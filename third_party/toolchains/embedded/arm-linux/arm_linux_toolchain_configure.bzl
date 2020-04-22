"""Repository rule for ARM cross compiler autoconfiguration."""

def _tpl(repository_ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl
    repository_ctx.template(
        out,
        Label("//third_party/toolchains/embedded/arm-linux:%s.tpl" % tpl),
        substitutions,
    )

def _arm_linux_toolchain_configure_impl(repository_ctx):
    _tpl(repository_ctx, "cc_config.bzl", {
        "%{AARCH64_COMPILER_PATH}%": str(repository_ctx.path(
            repository_ctx.attr.aarch64_repo,
        )),
        "%{ARMHF_COMPILER_PATH}%": str(repository_ctx.path(
            repository_ctx.attr.armhf_repo,
        )),
    })
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD")

arm_linux_toolchain_configure = repository_rule(
    implementation = _arm_linux_toolchain_configure_impl,
    attrs = {
        "aarch64_repo": attr.string(mandatory = True, default = ""),
        "armhf_repo": attr.string(mandatory = True, default = ""),
        "build_file": attr.label(),
    },
)
