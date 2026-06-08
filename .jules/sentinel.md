## 2025-05-14 - Path Traversal in SavedModel CLI Output
**Vulnerability:** Path traversal in `saved_model_cli.py`'s `run_saved_model_with_feed_dict` function.
**Learning:** The tool used output tensor names (keys) directly from the model signature as filenames. Since these keys are under the control of the model creator and not sanitized, a malicious model could cause the tool to write files to arbitrary locations on the filesystem using path traversal sequences like `../../`.
**Prevention:** Always sanitize or validate filenames derived from external data. Use `os.path.commonpath` with absolute paths to verify that the final path is contained within the intended base directory.
