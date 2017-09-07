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
  _tpl(repository_ctx, "CROSSTOOL", {
      "%{ARM_COMPILER_PATH}%": str(repository_ctx.path(
          repository_ctx.attr.remote_config_repo)),
  })
  repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")


arm_compiler_configure = repository_rule(
    implementation = _arm_compiler_configure_impl,
    attrs = {
        "remote_config_repo": attr.string(mandatory = False, default =""),
        "build_file": attr.label(),
    },
)
