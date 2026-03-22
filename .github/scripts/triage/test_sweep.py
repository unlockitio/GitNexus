"""Tests for sweep.py — all external calls (API, embedding) are mocked."""
from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from unittest.mock import patch, MagicMock, mock_open
from urllib.error import HTTPError

import numpy as np
import pytest

# Mock fastembed before importing sweep (which imports embedding_utils)
sys.modules["fastembed"] = MagicMock()

# Set required env vars before importing sweep (module-level constants read env)
os.environ.setdefault("GITHUB_TOKEN", "test-token")
os.environ.setdefault("GITHUB_REPOSITORY", "owner/repo")

from sweep import (
    github_api_get,
    fetch_all_open_items,
    fetch_repo_labels,
    apply_labels_to_item,
    generate_report,
    create_report_issue,
    write_report,
    main,
    TriageItem,
    RepoLabel,
    REPORT_FILE,
    REPORT_LABEL,
    API_PAGE_SIZE,
    MIN_SAMPLES_FOR_OUTLIER_DETECTION,
)


def _make_api_issue(number: int, title: str = "Test issue", is_pr: bool = False,
                    body: str = "Issue body", labels: list[str] | None = None) -> dict:
    """Helper to build a mock GitHub API issue response object."""
    result: dict = {
        "number": number,
        "title": title,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
        "body": body,
        "created_at": "2026-03-21T00:00:00Z",
        "labels": [{"name": lbl} for lbl in (labels or [])],
    }
    if is_pr:
        result["pull_request"] = {"url": "..."}
    return result


class TestGithubApiGet:
    """Tests for the github_api_get function."""

    @patch("sweep.urllib.request.urlopen")
    def test_successful_request(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([{"id": 1}]).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = github_api_get("/issues?state=open")
        assert result == [{"id": 1}]

    @patch("sweep.urllib.request.urlopen")
    def test_http_error_exits(self, mock_urlopen):
        error = HTTPError(
            url="https://api.github.com/repos/owner/repo/issues",
            code=403,
            msg="Forbidden",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b'{"message": "rate limited"}'),
        )
        mock_urlopen.side_effect = error

        with pytest.raises(SystemExit) as exc_info:
            github_api_get("/issues")
        assert exc_info.value.code == 1


class TestFetchAllOpenItems:
    """Tests for fetch_all_open_items."""

    @patch("sweep.github_api_get")
    def test_empty_repo(self, mock_get):
        mock_get.return_value = []
        items = fetch_all_open_items()
        assert items == []

    @patch("sweep.github_api_get")
    def test_single_page(self, mock_get):
        mock_get.return_value = [
            _make_api_issue(1, "Bug report"),
            _make_api_issue(2, "Feature request", is_pr=True),
        ]
        items = fetch_all_open_items()
        assert len(items) == 2
        assert items[0]["number"] == 1
        assert items[0]["is_pr"] is False
        assert items[1]["is_pr"] is True

    @patch("sweep.github_api_get")
    def test_text_field_constructed(self, mock_get):
        mock_get.return_value = [
            _make_api_issue(1, "My Title", body="My Body"),
        ]
        items = fetch_all_open_items()
        assert items[0]["text"] == "My Title\n\nMy Body"

    @patch("sweep.github_api_get")
    def test_null_body_handled(self, mock_get):
        issue = _make_api_issue(1, "No body")
        issue["body"] = None
        mock_get.return_value = [issue]
        items = fetch_all_open_items()
        assert items[0]["text"] == "No body\n\n"

    @patch("sweep.github_api_get")
    def test_labels_extracted(self, mock_get):
        mock_get.return_value = [
            _make_api_issue(1, "Labeled", labels=["bug", "high-priority"]),
        ]
        items = fetch_all_open_items()
        assert items[0]["labels"] == ["bug", "high-priority"]

    @patch("sweep.MAX_ITEMS", 3)
    @patch("sweep.github_api_get")
    def test_max_items_cap(self, mock_get):
        mock_get.return_value = [_make_api_issue(i) for i in range(100)]
        items = fetch_all_open_items()
        assert len(items) == 3

    @patch("sweep.API_PAGE_SIZE", 2)
    @patch("sweep.github_api_get")
    def test_pagination(self, mock_get):
        # First page: 2 items (full page), second page: 1 item (partial -> stop)
        mock_get.side_effect = [
            [_make_api_issue(1), _make_api_issue(2)],
            [_make_api_issue(3)],
        ]
        items = fetch_all_open_items()
        assert len(items) == 3
        assert mock_get.call_count == 2


class TestGenerateReport:
    """Tests for the markdown report generator."""

    def test_no_findings(self):
        items = [
            TriageItem(
                number=1, title="Test", html_url="https://example.com/1",
                is_pr=False, labels=[], created_at="2026-01-01", text="Test",
            ),
        ]
        report = generate_report(items, [], [])
        assert "## Triage Sweep Report" in report
        assert "Items analyzed:** 1" in report
        assert "None found." in report
        assert "0 outliers flagged" in report
        assert "0 duplicate pairs found" in report

    def test_with_outliers(self):
        items = [
            TriageItem(
                number=10, title="Spam Issue", html_url="https://example.com/10",
                is_pr=False, labels=[], created_at="2026-01-01", text="spam",
            ),
            TriageItem(
                number=20, title="Good Issue", html_url="https://example.com/20",
                is_pr=False, labels=[], created_at="2026-01-01", text="good",
            ),
        ]
        report = generate_report(items, [0], [])
        assert "#10" in report
        assert "Spam Issue" in report
        assert "1 outliers flagged" in report

    def test_with_duplicates(self):
        items = [
            TriageItem(
                number=1, title="First", html_url="https://example.com/1",
                is_pr=False, labels=[], created_at="2026-01-01", text="a",
            ),
            TriageItem(
                number=2, title="Second", html_url="https://example.com/2",
                is_pr=True, labels=[], created_at="2026-01-01", text="b",
            ),
        ]
        report = generate_report(items, [], [(0, 1, 0.954)])
        assert "#1" in report
        assert "#2" in report
        assert "0.954" in report
        assert "1 duplicate pairs found" in report

    def test_pr_type_label(self):
        items = [
            TriageItem(
                number=5, title="PR Title", html_url="https://example.com/5",
                is_pr=True, labels=[], created_at="2026-01-01", text="pr",
            ),
        ]
        report = generate_report(items, [0], [])
        assert "| PR |" in report

    def test_footer_present(self):
        items = [
            TriageItem(
                number=1, title="T", html_url="u",
                is_pr=False, labels=[], created_at="d", text="t",
            ),
        ]
        report = generate_report(items, [], [])
        assert "no LLM was used" in report


class TestCreateReportIssue:
    """Tests for creating the report GitHub issue."""

    @patch("sweep.urllib.request.urlopen")
    def test_successful_creation(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.read.return_value = json.dumps({
            "html_url": "https://github.com/owner/repo/issues/99",
        }).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Should not raise
        create_report_issue("# Test Report")

    @patch("sweep.urllib.request.urlopen")
    def test_http_error_exits(self, mock_urlopen):
        error = HTTPError(
            url="https://api.github.com/repos/owner/repo/issues",
            code=422,
            msg="Unprocessable",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b'{"message": "validation failed"}'),
        )
        mock_urlopen.side_effect = error

        with pytest.raises(SystemExit) as exc_info:
            create_report_issue("# Test Report")
        assert exc_info.value.code == 1


class TestWriteReport:
    """Tests for the write_report helper."""

    @patch("builtins.open", mock_open())
    def test_writes_to_file(self):
        write_report("# Report Content")
        from builtins import open as builtin_open  # noqa
        # Verify open was called with the right path
        from unittest.mock import call
        open_mock = open  # The patched version
        open_mock.assert_called_once_with(REPORT_FILE, "w", encoding="utf-8")  # type: ignore[attr-defined]
        open_mock().write.assert_called_once_with("# Report Content")  # type: ignore[attr-defined]


class TestFetchRepoLabels:
    """Tests for fetch_repo_labels."""

    @patch("sweep.github_api_get")
    def test_fetches_and_constructs_labels(self, mock_get):
        mock_get.return_value = [
            {"name": "bug", "description": "Something isn't working"},
            {"name": "enhancement", "description": "New feature or request"},
            {"name": "docs", "description": ""},
        ]
        labels = fetch_repo_labels()
        assert len(labels) == 3
        assert labels[0]["name"] == "bug"
        assert labels[0]["text"] == "bug: Something isn't working"
        assert labels[2]["text"] == "docs"  # no description, just name

    @patch("sweep.github_api_get")
    def test_empty_repo_labels(self, mock_get):
        mock_get.return_value = []
        labels = fetch_repo_labels()
        assert labels == []

    @patch("sweep.github_api_get")
    def test_null_description_handled(self, mock_get):
        mock_get.return_value = [
            {"name": "wontfix", "description": None},
        ]
        labels = fetch_repo_labels()
        assert labels[0]["text"] == "wontfix"


class TestApplyLabelsToItem:
    """Tests for apply_labels_to_item."""

    def test_empty_labels_skips(self):
        # Should not make any API call
        apply_labels_to_item(1, [])

    @patch("sweep.urllib.request.urlopen")
    def test_successful_label_application(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'[{"name": "bug"}]'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        # Should not raise
        apply_labels_to_item(42, ["bug", "enhancement"])

    @patch("sweep.urllib.request.urlopen")
    def test_http_error_is_non_fatal(self, mock_urlopen):
        error = HTTPError(
            url="https://api.github.com/repos/owner/repo/issues/1/labels",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=BytesIO(b'{"message": "not found"}'),
        )
        mock_urlopen.side_effect = error

        # Should NOT raise — labeling failures are warnings, not fatal
        apply_labels_to_item(1, ["bug"])


class TestGenerateReportWithLabels:
    """Tests for label suggestions in the report."""

    def test_report_includes_label_section(self):
        items = [
            TriageItem(
                number=1, title="Fix crash", html_url="https://example.com/1",
                is_pr=False, labels=[], created_at="2026-01-01", text="crash",
            ),
        ]
        suggestions = [[("bug", 0.85), ("enhancement", 0.42)]]
        report = generate_report(items, [], [], label_suggestions=suggestions)
        assert "Suggested Labels" in report
        assert "`bug` (0.85)" in report
        assert "1 items suggested for labeling" in report

    def test_report_skips_already_labeled_items(self):
        items = [
            TriageItem(
                number=1, title="Already labeled", html_url="https://example.com/1",
                is_pr=False, labels=["bug"], created_at="2026-01-01", text="bug",
            ),
        ]
        suggestions = [[("bug", 0.95)]]
        report = generate_report(items, [], [], label_suggestions=suggestions)
        assert "0 items suggested for labeling" in report
        assert "No unlabeled items" in report

    def test_report_excludes_outliers_from_suggestions(self):
        items = [
            TriageItem(
                number=1, title="Spam garbage", html_url="https://example.com/1",
                is_pr=False, labels=[], created_at="2026-01-01", text="spam",
            ),
            TriageItem(
                number=2, title="Real bug", html_url="https://example.com/2",
                is_pr=False, labels=[], created_at="2026-01-01", text="bug",
            ),
        ]
        suggestions = [[("bug", 0.85)], [("bug", 0.90)]]
        # Item 0 is an outlier — should be excluded from label suggestions
        report = generate_report(items, [0], [], label_suggestions=suggestions)
        assert "1 unlabeled items" in report  # only item 2
        assert "#2" in report
        # Item 1 (outlier) should NOT be in the suggestions table
        assert "Spam garbage" not in report.split("Suggested Labels")[1]

    def test_report_without_label_suggestions(self):
        items = [
            TriageItem(
                number=1, title="T", html_url="u",
                is_pr=False, labels=[], created_at="d", text="t",
            ),
        ]
        report = generate_report(items, [], [], label_suggestions=None)
        assert "Suggested Labels" not in report


class TestMain:
    """Tests for the main orchestration function."""

    @patch.dict(os.environ, {"GITHUB_TOKEN": "", "GITHUB_REPOSITORY": "owner/repo"})
    def test_missing_token_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch.dict(os.environ, {"GITHUB_TOKEN": "tok", "GITHUB_REPOSITORY": ""})
    def test_missing_repo_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    @patch("sweep.write_report")
    @patch("sweep.fetch_all_open_items", return_value=[])
    def test_no_items(self, mock_fetch, mock_write):
        main()
        mock_write.assert_called_once()
        report = mock_write.call_args[0][0]
        assert "No open issues or PRs found" in report

    @patch("sweep.create_report_issue")
    @patch("sweep.write_report")
    @patch("sweep.suggest_labels", return_value=[])
    @patch("sweep.find_duplicate_pairs", return_value=[])
    @patch("sweep.detect_outliers", return_value=[])
    @patch("sweep.reduce_dimensions")
    @patch("sweep.normalize_rows")
    @patch("sweep.embed_texts")
    @patch("sweep.fetch_repo_labels")
    @patch("sweep.fetch_all_open_items")
    def test_full_flow_with_enough_items(
        self, mock_fetch, mock_labels, mock_embed, mock_norm, mock_reduce,
        mock_outliers, mock_dupes, mock_suggest, mock_write, mock_create,
    ):
        """Test the full flow with >= MIN_SAMPLES items (outlier detection runs)."""
        items = [
            TriageItem(
                number=i, title=f"Item {i}", html_url=f"https://example.com/{i}",
                is_pr=False, labels=[], created_at="2026-01-01", text=f"text {i}",
            )
            for i in range(15)
        ]
        mock_fetch.return_value = items
        mock_labels.return_value = [
            RepoLabel(name="bug", description="Something broken", text="bug: Something broken"),
        ]

        embeddings = np.random.randn(15, 384).astype(np.float32)
        mock_embed.return_value = embeddings
        mock_norm.return_value = embeddings
        mock_reduce.return_value = np.random.randn(15, 10).astype(np.float32)

        main()

        mock_fetch.assert_called_once()
        mock_labels.assert_called_once()
        # embed_texts called twice: once for items, once for labels
        assert mock_embed.call_count == 2
        mock_norm.assert_called()
        mock_reduce.assert_called_once()
        mock_outliers.assert_called_once()
        mock_dupes.assert_called_once()
        mock_suggest.assert_called_once()
        mock_write.assert_called_once()
        mock_create.assert_called_once()

    @patch("sweep.create_report_issue")
    @patch("sweep.write_report")
    @patch("sweep.suggest_labels", return_value=[])
    @patch("sweep.find_duplicate_pairs", return_value=[])
    @patch("sweep.detect_outliers")
    @patch("sweep.reduce_dimensions")
    @patch("sweep.normalize_rows")
    @patch("sweep.embed_texts")
    @patch("sweep.fetch_repo_labels", return_value=[])
    @patch("sweep.fetch_all_open_items")
    def test_skips_outlier_detection_for_few_items(
        self, mock_fetch, mock_labels, mock_embed, mock_norm, mock_reduce,
        mock_outliers, mock_dupes, mock_suggest, mock_write, mock_create,
    ):
        """With < MIN_SAMPLES items, outlier detection should be skipped."""
        items = [
            TriageItem(
                number=i, title=f"Item {i}", html_url=f"https://example.com/{i}",
                is_pr=False, labels=[], created_at="2026-01-01", text=f"text {i}",
            )
            for i in range(5)
        ]
        mock_fetch.return_value = items

        embeddings = np.random.randn(5, 384).astype(np.float32)
        mock_embed.return_value = embeddings
        mock_norm.return_value = embeddings

        main()

        # Outlier detection should not have been called
        mock_reduce.assert_not_called()
        mock_outliers.assert_not_called()
        # But duplicates should still be checked
        mock_dupes.assert_called_once()

    @patch.dict(os.environ, {"INPUT_DRY_RUN": "true"})
    @patch("sweep.DRY_RUN", True)
    @patch("sweep.write_report")
    @patch("sweep.create_report_issue")
    @patch("sweep.apply_labels_to_item")
    @patch("sweep.suggest_labels", return_value=[[("bug", 0.85)]])
    @patch("sweep.find_duplicate_pairs", return_value=[])
    @patch("sweep.normalize_rows")
    @patch("sweep.embed_texts")
    @patch("sweep.fetch_repo_labels")
    @patch("sweep.fetch_all_open_items")
    def test_dry_run_skips_issue_creation_and_labeling(
        self, mock_fetch, mock_labels, mock_embed, mock_norm,
        mock_dupes, mock_suggest, mock_apply, mock_create, mock_write,
    ):
        items = [
            TriageItem(
                number=1, title="Item", html_url="https://example.com/1",
                is_pr=False, labels=[], created_at="2026-01-01", text="text",
            )
        ]
        mock_fetch.return_value = items
        mock_labels.return_value = [
            RepoLabel(name="bug", description="Broken", text="bug: Broken"),
        ]
        embeddings = np.random.randn(1, 384).astype(np.float32)
        mock_embed.return_value = embeddings
        mock_norm.return_value = embeddings

        main()

        mock_create.assert_not_called()
        mock_apply.assert_not_called()
        mock_write.assert_called_once()

    @patch("sweep.create_report_issue")
    @patch("sweep.write_report")
    @patch("sweep.apply_labels_to_item")
    @patch("sweep.suggest_labels")
    @patch("sweep.find_duplicate_pairs", return_value=[])
    @patch("sweep.normalize_rows")
    @patch("sweep.embed_texts")
    @patch("sweep.fetch_repo_labels")
    @patch("sweep.fetch_all_open_items")
    def test_applies_labels_to_unlabeled_items(
        self, mock_fetch, mock_labels, mock_embed, mock_norm,
        mock_dupes, mock_suggest, mock_apply, mock_write, mock_create,
    ):
        """When not dry run, top-1 label should be applied to unlabeled items."""
        items = [
            TriageItem(
                number=1, title="Crash bug", html_url="https://example.com/1",
                is_pr=False, labels=[], created_at="2026-01-01", text="crash",
            ),
            TriageItem(
                number=2, title="Already labeled", html_url="https://example.com/2",
                is_pr=False, labels=["enhancement"], created_at="2026-01-01", text="feat",
            ),
        ]
        mock_fetch.return_value = items
        mock_labels.return_value = [
            RepoLabel(name="bug", description="Broken", text="bug: Broken"),
        ]
        mock_suggest.return_value = [
            [("bug", 0.90)],       # item 1: unlabeled, should get labeled
            [("bug", 0.45)],       # item 2: already labeled, skip
        ]

        embeddings = np.random.randn(2, 384).astype(np.float32)
        mock_embed.return_value = embeddings
        mock_norm.return_value = embeddings

        main()

        # Only item 1 (unlabeled) should get a label applied
        mock_apply.assert_called_once_with(1, ["bug"])

    @patch("sweep.create_report_issue")
    @patch("sweep.write_report")
    @patch("sweep.apply_labels_to_item")
    @patch("sweep.suggest_labels")
    @patch("sweep.find_duplicate_pairs", return_value=[])
    @patch("sweep.detect_outliers")
    @patch("sweep.reduce_dimensions")
    @patch("sweep.normalize_rows")
    @patch("sweep.embed_texts")
    @patch("sweep.fetch_repo_labels")
    @patch("sweep.fetch_all_open_items")
    def test_outliers_do_not_get_labeled(
        self, mock_fetch, mock_labels, mock_embed, mock_norm, mock_reduce,
        mock_outliers, mock_dupes, mock_suggest, mock_apply, mock_write, mock_create,
    ):
        """Items flagged as outliers should not receive label suggestions."""
        items = [
            TriageItem(
                number=i, title=f"Item {i}", html_url=f"https://example.com/{i}",
                is_pr=False, labels=[], created_at="2026-01-01", text=f"text {i}",
            )
            for i in range(15)
        ]
        mock_fetch.return_value = items
        mock_labels.return_value = [
            RepoLabel(name="bug", description="Broken", text="bug: Broken"),
        ]
        # Outlier detection flags items 0 and 5
        mock_outliers.return_value = [0, 5]
        # Every item gets a suggestion
        mock_suggest.return_value = [[("bug", 0.85)] for _ in range(15)]

        embeddings = np.random.randn(15, 384).astype(np.float32)
        mock_embed.return_value = embeddings
        mock_norm.return_value = embeddings
        mock_reduce.return_value = np.random.randn(15, 10).astype(np.float32)

        main()

        # Items 0 and 5 are outliers — should NOT be labeled
        labeled_numbers = [call.args[0] for call in mock_apply.call_args_list]
        assert 0 not in labeled_numbers
        assert 5 not in labeled_numbers
        # Other items should be labeled (13 items: 15 total - 2 outliers)
        assert mock_apply.call_count == 13
