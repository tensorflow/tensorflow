"""Repository rule for Python autoconfiguration.
"""

load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load(
    "@xla//third_party/remote_config:common.bzl",
    "BAZEL_SH",
    "PYTHON_BIN_PATH",
    "PYTHON_LIB_PATH",
)

def _get_python_interpreter():
    python_toolchain_name = "python_{version}_host".format(
        version = HERMETIC_PYTHON_VERSION.replace(".", "_"),
    )
    return "@{}//:python".format(python_toolchain_name)

def _create_local_python_repository(repository_ctx):
    """Creates the repository containing files set up to build with Python."""

    platform_constraint = ""
    if repository_ctx.attr.platform_constraint:
        platform_constraint = "\"%s\"" % repository_ctx.attr.platform_constraint
    repository_ctx.template(
        "BUILD",
        repository_ctx.attr.build_tpl,
        {
            "%{PLATFORM_CONSTRAINT}": platform_constraint,
            "%{PYTHON_INTERPRETER}": repository_ctx.attr.python_interpreter,
        },
    )

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
        "build_tpl": attr.label(default = Label("//third_party/py:BUILD.tpl")),
        "python_interpreter": attr.string(default = _get_python_interpreter()),
    },
)

remote_python_configure = repository_rule(
    implementation = _create_local_python_repository,
    environ = _ENVIRONS,
    remotable = True,
    attrs = {
        "environ": attr.string_dict(),
        "platform_constraint": attr.string(),
        "build_tpl": attr.label(default = Label("//third_party/py:BUILD.tpl")),
        "python_interpreter": attr.string(default = _get_python_interpreter()),
    },
)

python_configure = repository_rule(
    implementation = _python_autoconf_impl,
    attrs = {
        "platform_constraint": attr.string(),
        "build_tpl": attr.label(default = Label("//third_party/py:BUILD.tpl")),
        "python_interpreter": attr.string(default = _get_python_interpreter()),
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
