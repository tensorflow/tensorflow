# -*- Python -*-
"""Repository rule for arm compiler autoconfiguration."""

_PYTHON_LIB_PATH = "PYTHON_LIB_PATH"

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl
  repository_ctx.template(
      out,
      Label("//third_party/toolchains/cpus/arm:%s.tpl" % tpl),
      substitutions)

def find_python_lib(repository_ctx):
  """Returns python path."""
  if _PYTHON_LIB_PATH in repository_ctx.os.environ:
    return repository_ctx.os.environ[_PYTHON_LIB_PATH].strip()
  fail("Environment variable PYTHON_LIB_PATH was not specified re-run ./configure")

def _arm_compiler_configure_impl(repository_ctx):
  python_lib_path = find_python_lib(repository_ctx)
  _tpl(repository_ctx, "CROSSTOOL", {
      "%{ARM_COMPILER_PATH}%": str(repository_ctx.path(
          repository_ctx.attr.remote_config_repo)),
      "%{PYTHON_LIB_PATH}%": python_lib_path,
  })
  repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")


arm_compiler_configure = repository_rule(
    implementation = _arm_compiler_configure_impl,
    attrs = {
        "remote_config_repo": attr.string(mandatory = False, default =""),
        "build_file": attr.label(),
    },
)
