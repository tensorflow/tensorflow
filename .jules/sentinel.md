## 2024-05-24 - TOC/TOU Weakness in Model Path Creation
**Vulnerability:** Use of `tempfile.mktemp()` for generating temporary directory paths (e.g., for `SavedModelBuilder`).
**Learning:** This codebase previously exhibited a pattern of using `mktemp` to generate a name, followed by an operation that implicitly expected to create a directory (like `SavedModelBuilder`). This hides a logic bug where a temporary directory should have been explicitly created, leading to a Time-of-Check to Time-of-Use (TOC/TOU) race condition.
**Prevention:** Always use `tempfile.mkdtemp()` when a temporary directory is needed, and `tempfile.mkstemp()` for files. Never use `tempfile.mktemp()`.
