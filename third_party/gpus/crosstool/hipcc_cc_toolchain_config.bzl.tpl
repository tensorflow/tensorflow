"""cc_toolchain_config rule for configuring rocm toolchain."""

load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "tool_path")

def _impl(ctx):
    toolchain_identifier = "local_linux"

    host_system_name = "local"

    target_system_name = "local"

    target_cpu = "local"

    target_libc = "local"

    compiler = "compiler"

    abi_version = "local"

    abi_libc_version = "local"

    cc_target_os = None

    builtin_sysroot = None

    action_configs = []

    features = []

    cxx_builtin_include_directories = ctx.attr.host_compiler_includes

    artifact_name_patterns = []

    make_variables = []

    tool_paths = [
        tool_path(name = "ar", path = "/usr/bin/ar"),
        tool_path(name = "compat-ld", path = "/usr/bin/ld"),
        tool_path(name = "cpp", path = "/usr/bin/cpp"),
        tool_path(name = "dwp", path = "/usr/bin/dwp"),
        tool_path(
            name = "gcc",
            path = "clang/bin/crosstool_wrapper_driver_rocm",
        ),
        tool_path(name = "gcov", path = "/usr/bin/gcov"),
        tool_path(name = "ld", path = "/usr/bin/ld"),
        tool_path(name = "nm", path = "/usr/bin/nm"),
        tool_path(name = "objcopy", path = "/usr/bin/objcopy"),
        tool_path(name = "objdump", path = "/usr/bin/objdump"),
        tool_path(name = "strip", path = "/usr/bin/strip"),
    ]

    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = artifact_name_patterns,
            cxx_builtin_include_directories = cxx_builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = make_variables,
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os,
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["local"]),
        "host_compiler_includes": attr.string_list(),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
