# This file is expanded from a template by rocm_configure.bzl
# Update rocm_configure.bzl#verify_build_defines when adding new variables.

load("@config_rocm_hipcc//rocm:build_defs.bzl", "hipcc_config")
load("@local_config_clang//:clang.bzl", "local_clang")
load(":cc_toolchain_config.bzl", "cc_toolchain_config")

# Local clang configuration for non-hermetic toolchain
_LOCAL_CLANG = local_clang()

# ROCm configuration from hermetic hipcc
_HIPCC_CONFIG = hipcc_config()

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

toolchain(
    name = "toolchain-linux-x86_64",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "arm": ":cc-compiler-local",
        "aarch64": ":cc-compiler-local",
        "k8": ":cc-compiler-local",
        "piii": ":cc-compiler-local",
        "ppc": ":cc-compiler-local",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":crosstool_wrapper_driver_is_not_gcc",
    ar_files = ":crosstool_wrapper_driver_is_not_gcc",
    as_files = ":crosstool_wrapper_driver_is_not_gcc",
    compiler_files = ":crosstool_wrapper_driver_is_not_gcc",
    dwp_files = ":empty",
    linker_files = ":crosstool_wrapper_driver_is_not_gcc",
    objcopy_files = ":empty",
    strip_files = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
    toolchain_config = ":cc-compiler-local-config",
    toolchain_identifier = "local_linux",
)

cc_toolchain_config(
    name = "cc-compiler-local-config",
    abi_libc_version = "local",
    abi_version = "local",
    # Compiler path from local_clang_info(), sets CLANG_COMPILER_PATH env var
    clang_compiler_path = _LOCAL_CLANG.compiler_path,
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        "-no-canonical-prefixes",
    ],
    compiler = "compiler",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "local",
    # Include directories detected from local clang + ROCm includes
    cxx_builtin_include_directories = _LOCAL_CLANG.include_directories,
    cxx_flags = ["-std=c++17"],
    dbg_compile_flags = ["-g"],
    host_compiler_path = "clang/bin/crosstool_wrapper_driver_is_not_gcc",
    host_compiler_prefix = "/usr/bin",
    host_system_name = "local",
    link_flags = [
        "-fuse-ld=lld",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-Wl,--allow-shlib-undefined",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    linker_bin_path = "external/config_rocm_hipcc/rocm/" + _HIPCC_CONFIG.rocm_root + "/bin",
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = [],
    supports_start_end_lib = True,
    target_libc = "local",
    target_system_name = "local",
    toolchain_identifier = "local_linux",
    unfiltered_compile_flags = [
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "crosstool_wrapper_driver_is_not_gcc",
    srcs = [
        ":clang/bin/crosstool_wrapper_driver_is_not_gcc",
        "@config_rocm_hipcc//rocm:toolchain_data",
    ],
)
