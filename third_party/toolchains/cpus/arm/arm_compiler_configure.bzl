# -*- Python -*-
"""Repository rule for arm compiler autoconfiguration."""

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl
  repository_ctx.template(
      out,
      Label("//third_party/toolchains/cpus/arm:%s.tpl" % tpl),
      substitutions)

def _get_python_bin(repository_ctx):
  """Gets the python bin path."""
  python_bin = _get_env_var(repository_ctx, _PYTHON_BIN_PATH,
                            None, False)
  if python_bin != None:
    return python_bin
  python_bin_path = repository_ctx.which("python")
  if python_bin_path != None:
    return str(python_bin_path)
  path = _get_env_var(repository_ctx, "PATH")
  _python_configure_fail("Cannot find python in PATH, please make sure " +
      "python is installed and add its directory in PATH, or set the " +
      "environment variable PYTHON_BIN_PATH.\nPATH=%s" % (path))

def _get_python_include(repository_ctx):
  """Gets the python include path."""
  python_bin = _get_python_bin(repository_ctx)
  result = _execute(repository_ctx,
                    [python_bin, "-c",
                     'from __future__ import print_function;' +
                     'from distutils import sysconfig;' +
                     'print(sysconfig.get_python_inc())'],
                    error_msg="Problem getting python include path.",
                    error_details=("Is the Python binary path set up right? " +
                                   "(See ./configure or PYTHON_BIN_PATH.) " +
                                   "Is distutils installed?"))
  return result.stdout.splitlines()[0]

def _arm_compiler_configure_impl(repository_ctx):
  python_include_path = _get_python_include(repository_ctx)
  _tpl(repository_ctx, "CROSSTOOL", {
      "%{ARM_COMPILER_PATH}%": str(repository_ctx.path(
          repository_ctx.attr.remote_config_repo)),
      "%{PYTHON_INCLUDE_PATH}%": python_include_path,
  })
  repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")


arm_compiler_configure = repository_rule(
    implementation = _arm_compiler_configure_impl,
    attrs = {
        "remote_config_repo": attr.string(mandatory = False, default =""),
        "build_file": attr.label(),
    },
)
