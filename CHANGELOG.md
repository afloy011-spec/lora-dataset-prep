# Changelog

All notable changes to **lora-dataset-prep** are documented here.
Versions follow [SemVer](https://semver.org); each release is a git tag (`vX.Y.Z`).

## [2.3.0] — 2026-07-02

### Fixed
- **EXIF orientation is baked into the pixels on `--resize` and WebP
  conversion.** Phone photos are often stored rotated with only an EXIF flag
  set; training pipelines read raw pixels and ignore the flag, so such images
  silently trained sideways. `--resize` (and WebP→PNG conversion) now apply
  `exif_transpose` first; upright images are still byte-for-byte copied with
  no re-encode.
- A caption with no trigger token at all now produces one warning instead of
  two stacked ones ("missing" + "does not start with").

### Added
- **EXIF-rotation detection** in the quality gate and in `--validate-only`:
  rotated images are flagged with a hint to pass `--resize`.
- **`--recursive`** — scan the source folder including subfolders (hidden
  folders are skipped). `raw/` backup names are flattened
  (`sub__01.jpg`) so same-named files from different subfolders can't collide.
  Without the flag, a note now reports how many subfolders were ignored.
- **Fail-fast VLM auth check**: `--captions vlm --execute` verifies Anthropic
  credentials up front (one clear error) instead of failing once per image;
  a dry-run only warns.
- **CI** — GitHub Actions runs the pytest suite on Python 3.9–3.12 for every
  push and pull request (79 tests).
- This changelog.

### Changed
- A trigger token containing `{` or `}` is now a hard error (it is inserted
  into caption templates via `str.format()` and crashed mid-run).
- `--move` prints an explicit deletion warning in the dry-run plan, extra-loud
  when combined with `--no-backup`.

## [2.2.0] — 2026-06-11

Initial public release: scan → quality gate (dimensions, blur, SHA-256 exact
dupes, dHash near-dupes) → organize with sequential names + `raw/` backup →
caption (template / minimal / Claude VLM / local Florence-2 or BLIP) →
validate (20+ checks incl. attribute-sticking detection) → dataset card +
`metadata.json`. Dry-run by default. Ships as a Claude Code / Cursor agent
skill (`SKILL.md` + `reference.md`).
