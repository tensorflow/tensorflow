"""
Repository rule to set python version.
Can be set via build parameter "--repo_env=TF_PYTHON_VERSION=3.10"
Defaults to 3.10.
"""

VERSIONS = ["3.9", "3.10", "3.11", "3.12"]
DEFAULT_VERSION = "3.11"
WARNING = """
TF_PYTHON_VERSION environment variable was not set correctly; using Python {}.

To set Python version, run:
export TF_PYTHON_VERSION=3.11
""".format(DEFAULT_VERSION)

content = """
TF_PYTHON_VERSION = "{}"
HERMETIC_PYTHON_VERSION = "{}"
"""

def _python_repository_impl(repository_ctx):
    repository_ctx.file("BUILD", "")
    version = repository_ctx.os.environ.get("TF_PYTHON_VERSION", "")
    if version not in VERSIONS:
        print(WARNING)  # buildifier: disable=print
        version = DEFAULT_VERSION
    repository_ctx.file(
        "py_version.bzl",
        content.format(version, version),
    )

python_repository = repository_rule(
    implementation = _python_repository_impl,
    environ = ["TF_PYTHON_VERSION"],
)
