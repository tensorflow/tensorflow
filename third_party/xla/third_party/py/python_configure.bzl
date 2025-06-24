"""Repository rule for Python autoconfiguration.
"""

load(
    "@local_xla//third_party/remote_config:common.bzl",
    "BAZEL_SH",
    "PYTHON_BIN_PATH",
    "PYTHON_LIB_PATH",
)

def _create_local_python_repository(repository_ctx):
    """Creates the repository containing files set up to build with Python."""

    # Resolve all labels before doing any real work. Resolving causes the
    # function to be restarted with all previous state being lost. This
    # can easily lead to a O(n^2) runtime in the number of labels.
    build_tpl = repository_ctx.path(Label("//third_party/py:BUILD.tpl"))
    platform_constraint = ""
    if repository_ctx.attr.platform_constraint:
        platform_constraint = "\"%s\"" % repository_ctx.attr.platform_constraint
    repository_ctx.template("BUILD", build_tpl, {"%{PLATFORM_CONSTRAINT}": platform_constraint})

def _python_autoconf_impl(repository_ctx):
    """Implementation of the python_autoconf repository rule."""
    _create_local_python_repository(repository_ctx)

_ENVIRONS = [
    BAZEL_SH,
    PYTHON_BIN_PATH,
    PYTHON_LIB_PATH,
]

local_python_configure = repository_rule(
    implementation = _create_local_python_repository,
    attrs = {
        "environ": attr.string_dict(),
        "platform_constraint": attr.string(),
    },
)

remote_python_configure = repository_rule(
    implementation = _create_local_python_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "platform_constraint": attr.string(),
    },
)

python_configure = repository_rule(
    implementation = _python_autoconf_impl,
    attrs = {
        "platform_constraint": attr.string(),
    },
)
"""Detects and configures the local Python.

Add the following to your WORKSPACE FILE:

```python
python_configure(name = "local_config_python")
```

Args:
  name: A unique name for this workspace rule.
"""
