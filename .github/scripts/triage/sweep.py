"""Triage sweep: fetch open issues/PRs, detect outliers and duplicates, generate a report.

Entrypoint script for the triage-sweep workflow. Fetches all open items via
the GitHub REST API, delegates embedding and analysis to embedding_utils,
generates a markdown report, and optionally creates a report issue.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.parse
from typing import TypedDict
from datetime import datetime, timezone

from embedding_utils import (
    embed_texts,
    normalize_rows,
    reduce_dimensions,
    detect_outliers,
    find_duplicate_pairs,
    suggest_labels,
)

# ── Thresholds (overridable via workflow_dispatch inputs) ──────────────

# Mahalanobis distance beyond which an item is flagged as an outlier.
# Default 3.0 ~ 99.7% of a Gaussian distribution (3-sigma rule).
MAHALANOBIS_THRESHOLD: float = float(os.environ.get("INPUT_MAHALANOBIS_THRESHOLD", "3.0"))

# Cosine similarity above which two items are flagged as duplicates.
# 0.92 catches near-identical issues while tolerating paraphrasing.
COSINE_THRESHOLD: float = float(os.environ.get("INPUT_COSINE_THRESHOLD", "0.92"))

# Hard cap on items to process. Prevents runaway costs on very large repos.
MAX_ITEMS: int = int(os.environ.get("INPUT_MAX_ITEMS", "500"))

# When true, print report to stdout/file but do not create a GitHub issue.
DRY_RUN: bool = os.environ.get("INPUT_DRY_RUN", "false").lower() == "true"

# ── Fixed constants (not user-configurable) ───────────────────────────

# Minimum number of samples required for EllipticEnvelope to fit
# a Gaussian. Below this, outlier detection is skipped because
# covariance estimation is unreliable.
MIN_SAMPLES_FOR_OUTLIER_DETECTION: int = 10

# PCA: retain components explaining this fraction of variance.
# 0.95 keeps 95% of information while reducing dimensionality enough
# for EllipticEnvelope to be numerically stable.
PCA_VARIANCE_RATIO: float = 0.95

# PCA: maximum number of components regardless of variance ratio.
# Caps dimensionality for EllipticEnvelope's n_samples > n_features^2 rule.
PCA_MAX_COMPONENTS: int = 50

# GitHub REST API page size (max allowed is 100).
API_PAGE_SIZE: int = 100

# Report issue label.
REPORT_LABEL: str = "triage-report"

# Report file path (written for the summary step to pick up).
REPORT_FILE: str = "/tmp/triage-report.md"


class TriageItem(TypedDict):
    """One open issue or PR, with only the fields we need."""
    number: int
    title: str
    html_url: str
    is_pr: bool
    labels: list[str]
    created_at: str
    # title + body concatenated, used as embedding input
    text: str


def github_api_get(path: str) -> list[dict]:
    """Make a single authenticated GET request to the GitHub REST API.

    Reads GITHUB_TOKEN and GITHUB_REPOSITORY from env. Raises SystemExit
    with the HTTP status and response body on any non-2xx response.
    """
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}{path}"

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"::error::GitHub API {e.code}: {body}")
        sys.exit(1)


def fetch_all_open_items() -> list[TriageItem]:
    """Paginate through all open issues and PRs.

    Returns up to MAX_ITEMS TriageItem dicts. Items with a pull_request
    key are marked is_pr=True. The text field is title + body concatenated.
    """
    items: list[TriageItem] = []
    page = 1

    while len(items) < MAX_ITEMS:
        path = (
            f"/issues?state=open&per_page={API_PAGE_SIZE}"
            f"&sort=created&direction=desc&page={page}"
        )
        data = github_api_get(path)

        if not data:
            break

        for raw in data:
            if len(items) >= MAX_ITEMS:
                break

            body = raw.get("body", "") or ""
            items.append(TriageItem(
                number=raw["number"],
                title=raw["title"],
                html_url=raw["html_url"],
                is_pr="pull_request" in raw,
                labels=[lbl["name"] for lbl in raw.get("labels", [])],
                created_at=raw["created_at"],
                text=f"{raw['title']}\n\n{body}",
            ))

        if len(data) < API_PAGE_SIZE:
            break

        page += 1

    return items


class RepoLabel(TypedDict):
    """A label from the repo with its embedding text."""
    name: str
    description: str
    # "name: description" concatenated for embedding
    text: str


def fetch_repo_labels() -> list[RepoLabel]:
    """Fetch all labels from the repository.

    Returns labels with name, description, and a text field suitable
    for embedding ("name: description"). Labels with no description
    use just the name.
    """
    data = github_api_get("/labels?per_page=100")
    labels: list[RepoLabel] = []
    for raw in data:
        name = raw["name"]
        desc = raw.get("description", "") or ""
        text = f"{name}: {desc}" if desc else name
        labels.append(RepoLabel(name=name, description=desc, text=text))
    return labels


def apply_labels_to_item(item_number: int, labels: list[str]) -> None:
    """Add labels to a single issue/PR via the GitHub API.

    Skips silently if labels list is empty. Uses POST which adds labels
    without removing existing ones.
    """
    if not labels:
        return

    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}/issues/{item_number}/labels"

    payload = json.dumps({"labels": labels}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # Non-fatal: log warning but don't abort the sweep
        print(f"::warning::Failed to label #{item_number}: {e.code} {body}")


def generate_report(
    items: list[TriageItem],
    outlier_indices: list[int],
    duplicate_pairs: list[tuple[int, int, float]],
    label_suggestions: list[list[tuple[str, float]]] | None = None,
) -> str:
    """Generate a structured markdown triage report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    repo = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")

    lines: list[str] = [
        "## Triage Sweep Report",
        "",
        f"**Run:** {now} UTC",
        f"**Items analyzed:** {len(items)}",
        f"**Thresholds:** Mahalanobis > {MAHALANOBIS_THRESHOLD}, Cosine > {COSINE_THRESHOLD}",
        "",
        f"### Potential Outliers / Spam ({len(outlier_indices)})",
        "",
        "Items with unusually high Mahalanobis distance from the distribution center.",
        "These may be spam, off-topic, or poorly described.",
        "",
    ]

    if outlier_indices:
        lines.append("| # | Type | Title | Distance |")
        lines.append("|---|------|-------|----------|")
        for idx in outlier_indices:
            item = items[idx]
            kind = "PR" if item["is_pr"] else "Issue"
            lines.append(
                f"| [#{item['number']}]({item['html_url']}) "
                f"| {kind} | {item['title']} | flagged |"
            )
    else:
        lines.append("None found.")

    lines.extend([
        "",
        f"### Potential Duplicates ({len(duplicate_pairs)} pairs)",
        "",
        "Pairs of items with cosine similarity above the threshold.",
        "",
    ])

    if duplicate_pairs:
        lines.append("| Item A | Item B | Similarity |")
        lines.append("|--------|--------|------------|")
        for i, j, sim in duplicate_pairs:
            a = items[i]
            b = items[j]
            kind_a = "PR" if a["is_pr"] else "Issue"
            kind_b = "PR" if b["is_pr"] else "Issue"
            lines.append(
                f"| [#{a['number']}]({a['html_url']}) {kind_a}: {a['title']} "
                f"| [#{b['number']}]({b['html_url']}) {kind_b}: {b['title']} "
                f"| {sim:.3f} |"
            )
    else:
        lines.append("None found.")

    # ── Label suggestions section ────────────────────────────────────
    outlier_set = set(outlier_indices)
    if label_suggestions is not None:
        # Only unlabeled, non-outlier items — spam shouldn't get categorized
        items_with_suggestions = [
            (i, sugs) for i, sugs in enumerate(label_suggestions)
            if sugs and not items[i]["labels"] and i not in outlier_set
        ]
        lines.extend([
            "",
            f"### Suggested Labels ({len(items_with_suggestions)} unlabeled items)",
            "",
            "Labels suggested by embedding similarity against repo label descriptions.",
            "Only shown for unlabeled items that were not flagged as outliers.",
            "",
        ])

        if items_with_suggestions:
            lines.append("| # | Type | Title | Suggested Labels |")
            lines.append("|---|------|-------|-----------------|")
            for idx, sugs in items_with_suggestions:
                item = items[idx]
                kind = "PR" if item["is_pr"] else "Issue"
                label_strs = [f"`{name}` ({score:.2f})" for name, score in sugs]
                lines.append(
                    f"| [#{item['number']}]({item['html_url']}) "
                    f"| {kind} | {item['title']} | {', '.join(label_strs)} |"
                )
        else:
            lines.append("No unlabeled items need suggestions.")

    lines.extend([
        "",
        "### Summary",
        "",
        f"- {len(outlier_indices)} outliers flagged for review",
        f"- {len(duplicate_pairs)} duplicate pairs found",
        f"- {len(items)} items analyzed in total",
    ])

    if label_suggestions is not None:
        applied = sum(
            1 for i, s in enumerate(label_suggestions)
            if s and not items[i]["labels"] and i not in outlier_set
        )
        lines.append(f"- {applied} items suggested for labeling")

    lines.extend([
        "",
        "---",
        f"*Generated by [triage-sweep](https://github.com/{repo}/actions) — no LLM was used.*",
    ])

    return "\n".join(lines)


def create_report_issue(report_body: str) -> None:
    """Create a GitHub issue with the triage report.

    Posts to the issues API with the triage-report label.
    Raises SystemExit on non-201 response.
    """
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repo}/issues"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = json.dumps({
        "title": f"Triage Sweep Report — {today}",
        "body": report_body,
        "labels": [REPORT_LABEL],
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
            if resp.status != 201:
                print(f"::error::Failed to create issue: {resp.status} {resp_body}")
                sys.exit(1)
            result = json.loads(resp_body)
            print(f"Created issue: {result.get('html_url', 'unknown')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"::error::Failed to create issue: {e.code} {body}")
        sys.exit(1)


def write_report(report: str) -> None:
    """Write the report to the file system for the summary step."""
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)


def main() -> None:
    """Orchestrate the full triage sweep."""
    # 1. Validate environment
    for var in ("GITHUB_TOKEN", "GITHUB_REPOSITORY"):
        if not os.environ.get(var):
            print(f"::error::Missing required environment variable: {var}")
            sys.exit(1)

    # 2. Fetch all open issues + PRs
    items = fetch_all_open_items()
    print(f"Fetched {len(items)} open items")

    if len(items) == 0:
        report = "## Triage Sweep Report\n\nNo open issues or PRs found."
        write_report(report)
        print("No items to analyze.")
        return

    # 3. Extract texts for embedding
    texts: list[str] = [item["text"] for item in items]

    # 4. Embed all texts (returns numpy float32 array of shape [n, 384])
    embeddings = embed_texts(texts)

    # 5. L2-normalize
    embeddings = normalize_rows(embeddings)

    # 6. Outlier detection (Mahalanobis via EllipticEnvelope)
    outlier_indices: list[int] = []
    if len(items) >= MIN_SAMPLES_FOR_OUTLIER_DETECTION:
        reduced = reduce_dimensions(embeddings, PCA_VARIANCE_RATIO, PCA_MAX_COMPONENTS)
        outlier_indices = detect_outliers(reduced, MAHALANOBIS_THRESHOLD)
    else:
        print(
            f"Skipping outlier detection: {len(items)} items < "
            f"{MIN_SAMPLES_FOR_OUTLIER_DETECTION} minimum"
        )

    # 7. Duplicate detection (pairwise cosine similarity)
    duplicate_pairs = find_duplicate_pairs(embeddings, COSINE_THRESHOLD)

    # 8. Label suggestion via embedding similarity
    label_suggestions: list[list[tuple[str, float]]] | None = None
    repo_labels = fetch_repo_labels()
    if repo_labels:
        label_texts = [lbl["text"] for lbl in repo_labels]
        label_names = [lbl["name"] for lbl in repo_labels]
        label_embeddings = embed_texts(label_texts)
        label_embeddings = normalize_rows(label_embeddings)
        label_suggestions = suggest_labels(embeddings, label_embeddings, label_names)
        print(f"Computed label suggestions against {len(repo_labels)} repo labels")

        # Apply top label to unlabeled items (unless dry run)
        # Skip outliers — flagged items shouldn't get categorized
        outlier_set = set(outlier_indices)
        if not DRY_RUN:
            applied_count = 0
            for i, sugs in enumerate(label_suggestions):
                if sugs and not items[i]["labels"] and i not in outlier_set:
                    # Apply only the top-1 label (highest confidence)
                    apply_labels_to_item(items[i]["number"], [sugs[0][0]])
                    applied_count += 1
            print(f"Applied labels to {applied_count} unlabeled items")
    else:
        print("No repo labels found — skipping label suggestions")

    # 9. Generate report
    report = generate_report(items, outlier_indices, duplicate_pairs, label_suggestions)

    # 10. Write report to file (for summary step)
    write_report(report)

    # 11. Create report issue (unless dry run)
    if DRY_RUN:
        print("Dry run — skipping issue creation and label application.")
        print(report)
    else:
        create_report_issue(report)
        print("Report issue created.")


if __name__ == "__main__":
    main()
