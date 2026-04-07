import * as path from 'path';

/**
 * Recursively convert absolute filePath values to paths relative to basePath.
 *
 * Handles:
 * - `filePath` and `file_path` keys (object properties)
 * - Nested arrays and objects
 * - Null/undefined values (skipped safely)
 * - Cross-platform path separators (normalizes to forward slash)
 *
 * Only transforms if:
 * - Value is a string
 * - Value is an absolute path
 * - Value starts with basePath (with separator)
 */
export function relativizeFilePaths(obj: unknown, basePath: string): void {
  if (obj === null || obj === undefined) return;
  if (typeof obj === 'string') return;

  if (Array.isArray(obj)) {
    for (const item of obj) {
      relativizeFilePaths(item, basePath);
    }
    return;
  }

  if (typeof obj !== 'object') return;

  const sep = path.sep;
  const normalizedBase = basePath.endsWith(sep) ? basePath : basePath + sep;

  for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
    if (
      (key === 'filePath' || key === 'file_path' || key === 'handlerFile' || key === 'consumerFile' || key === 'epFilePath') &&
      typeof value === 'string' &&
      path.isAbsolute(value)
    ) {
      if (value.startsWith(normalizedBase) || value.startsWith(basePath + '/')) {
        const relative = path.relative(basePath, value);
        (obj as Record<string, unknown>)[key] = relative.split(path.sep).join('/');
      }
    } else {
      relativizeFilePaths(value, basePath);
    }
  }
}
