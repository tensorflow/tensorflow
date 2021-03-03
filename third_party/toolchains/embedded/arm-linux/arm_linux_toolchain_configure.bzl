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
    # We need to find a cross-compilation include directory for Python, so look
    # for an environment variable. Be warned, this crosstool template is only
    # regenerated on the first run of Bazel, so if you change the variable after
    # it may not be reflected in later builds. Doing a shutdown and clean of Bazel
    # doesn't fix this, you'll need to delete the generated file at something like:
    # external/local_config_arm_compiler/CROSSTOOL in your Bazel install.
    if "CROSSTOOL_PYTHON_INCLUDE_PATH" in repository_ctx.os.environ:
        python_include_path = repository_ctx.os.environ["CROSSTOOL_PYTHON_INCLUDE_PATH"]
    else:
        python_include_path = "/usr/include/python3.5"
    _tpl(repository_ctx, "cc_config.bzl", {
        "%{AARCH64_COMPILER_PATH}%": str(repository_ctx.path(
            repository_ctx.attr.aarch64_repo,
        )),
        "%{ARMHF_COMPILER_PATH}%": str(repository_ctx.path(
            repository_ctx.attr.armhf_repo,
        )),
        "%{PYTHON_INCLUDE_PATH}%": python_include_path,
    })
    repository_ctx.symlink(Label(repository_ctx.attr.build_file), "BUILD")

arm_linux_toolchain_configure = repository_rule(
    implementation = _arm_linux_toolchain_configure_impl,
    attrs = {
        "aarch64_repo": attr.string(mandatory = True, default = ""),
        "armhf_repo": attr.string(mandatory = True, default = ""),
        "build_file": attr.string(),
    },
)
