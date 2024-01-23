"""
Repository rule to set python version and wheel name.

Version can be set via build parameter "--repo_env=TF_PYTHON_VERSION=3.10"
Defaults to 3.10.

To set wheel name, add "--repo_env=WHEEL_NAME=tensorflow_cpu"
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
WHEEL_NAME = "{}"
WHEEL_COLLAB = "{}"
"""

def _python_repository_impl(repository_ctx):
    repository_ctx.file("BUILD", "")
    version = repository_ctx.os.environ.get("TF_PYTHON_VERSION", "")
    wheel_name = repository_ctx.os.environ.get("WHEEL_NAME", "tensorflow")
    wheel_collab = repository_ctx.os.environ.get("WHEEL_COLLAB", False)
    if version not in VERSIONS:
        print(WARNING)  # buildifier: disable=print
        version = DEFAULT_VERSION
    repository_ctx.file(
        "py_version.bzl",
        content.format(version, version, wheel_name, wheel_collab),
    )

python_repository = repository_rule(
    implementation = _python_repository_impl,
    environ = ["TF_PYTHON_VERSION", "WHEEL_NAME", "WHEEL_COLLAB"],
)
