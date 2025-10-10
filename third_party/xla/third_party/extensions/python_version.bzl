"""
A much simplified version of third_party/py/python_repo.bzl, which generates the "python_version_repo" repo.

This is just to keep the current build compatible with both WORKSPACE and Bzlmod, we may not need this in future.
"""

_PY_VERSION_BZL = """
HERMETIC_PYTHON_VERSION = "{version}"
USE_PYWRAP_RULES = {use_pywrap_rules}
"""

def _python_version_repo_impl(repository_ctx):
    version = repository_ctx.os.environ.get("HERMETIC_PYTHON_VERSION", "3.11")
    use_pywrap_rules = bool(
        repository_ctx.os.environ.get("USE_PYWRAP_RULES", False),
    )
    repository_ctx.file("BUILD.bazel", "")
    repository_ctx.file(
        "py_version.bzl",
        _PY_VERSION_BZL.format(
            version = version,
            use_pywrap_rules = use_pywrap_rules,
        ),
    )

python_version_repo = repository_rule(
    implementation = _python_version_repo_impl,
    environ = [
        "HERMETIC_PYTHON_VERSION",
        "USE_PYWRAP_RULES",
    ],
)

python_version_ext = module_extension(
    implementation = lambda mctx: python_version_repo(name = "python_version_repo"),
)
