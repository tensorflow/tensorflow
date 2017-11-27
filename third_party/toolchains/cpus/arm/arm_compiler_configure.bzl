# -*- Python -*-
"""Repository rule for arm compiler autoconfiguration."""

def _tpl(repository_ctx, tpl, substitutions={}, out=None):
  if not out:
    out = tpl
  repository_ctx.template(
      out,
      Label("//third_party/toolchains/cpus/arm:%s.tpl" % tpl),
      substitutions)


def _arm_compiler_configure_impl(repository_ctx):
  # We need to find a cross-compilation include directory for Python, so look
  # for an environment variable. Be warned, this crosstool template is only
  # regenerated on the first run of Bazel, so if you change the variable after
  # it may not be reflected in later builds. Doing a shutdown and clean of Bazel
  # doesn't fix this, you'll need to delete the generated file at something like:
  # external/local_config_arm_compiler/CROSSTOOL in your Bazel install.
  if "CROSSTOOL_PYTHON_INCLUDE_PATH" in repository_ctx.os.environ:
    python_include_path = repository_ctx.os.environ["CROSSTOOL_PYTHON_INCLUDE_PATH"]
  else:
    python_include_path = "/usr/include/python2.7"
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
