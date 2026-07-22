## 2024-05-24 - TOC/TOU Weakness in Model Path Creation
**Vulnerability:** Use of `tempfile.mktemp()` for generating temporary directory paths (e.g., for `SavedModelBuilder`).
**Learning:** This codebase previously exhibited a pattern of using `mktemp` to generate a name, followed by an operation that implicitly expected to create a directory (like `SavedModelBuilder`). This hides a logic bug where a temporary directory should have been explicitly created, leading to a Time-of-Check to Time-of-Use (TOC/TOU) race condition.
**Prevention:** Always use `tempfile.mkdtemp()` when a temporary directory is needed, and `tempfile.mkstemp()` for files. Never use `tempfile.mktemp()`.

## 2024-05-24 - Resource Leak Prevention with mkdtemp
**Vulnerability:** Resource Leak (Disk Space Exhaustion)
**Learning:** Fixing a TOC/TOU vulnerability by switching from `mktemp` to `mkdtemp` creates physical directories on disk. If these are not cleaned up (especially in test suites), they can accumulate and exhaust disk space on CI/CD environments or developer machines.
**Prevention:** When using `mkdtemp`, ensure proper cleanup is implemented. In scripts or tests without explicit teardown, use `atexit.register(shutil.rmtree, path, ignore_errors=True)`.

## 2024-05-24 - Cross-Platform Cleanup Handlers
**Vulnerability:** Resource Leak/Build Failure (Disk Space Exhaustion and CI flakiness)
**Learning:** While using `shutil.rmtree` is standard in Python, it often fails on Windows environments (causing exit code 34) due to file locking when tearing down temporary directories created with `mkdtemp`.
**Prevention:** In TensorFlow, use `tensorflow.python.lib.io.file_io.delete_recursively` with error handling (`except tf.errors.OpError`) for cross-platform robustness when cleaning up temporary directories in an `atexit` hook.
