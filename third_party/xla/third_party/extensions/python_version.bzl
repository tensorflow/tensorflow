"""
A much simplified version of third_party/py/python_repo.bzl, which generates the "python_version_repo" repo.

This is just to keep the current build compatible with both WORKSPACE and Bzlmod, we may not need this in future.
"""

_PY_VERSION_BZL = """
HERMETIC_PYTHON_VERSION = "{version}"
HERMETIC_PYTHON_VERSION_KIND = "{py_kind}"
USE_PYWRAP_RULES = {use_pywrap_rules}
# TODO(pcloudy): Figure out how to support requirements_lock in Bzlmod.
REQUIREMENTS = "//:requirements.txt"
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
            py_kind = "",  # TODO(pcloudy): introduce this value properly.
            use_pywrap_rules = use_pywrap_rules,
        ),
    )

python_version_repo = repository_rule(
    implementation = _python_version_repo_impl,
    environ = [
        "HERMETIC_PYTHON_VERSION",
        "HERMETIC_PYTHON_VERSION_KIND",
        "USE_PYWRAP_RULES",
    ],
)

python_version_ext = module_extension(
    implementation = lambda mctx: python_version_repo(name = "python_version_repo"),
)
