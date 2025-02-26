# This file is expanded from a template by cuda_configure.bzl
# Update cuda_configure.bzl#verify_build_defines when adding new variables.

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

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
    compiler_files = ":crosstool_wrapper_driver_is_not_gcc",
    ar_files = ":crosstool_wrapper_driver_is_not_gcc",
    as_files = ":crosstool_wrapper_driver_is_not_gcc",
    dwp_files = ":empty",
    linker_files = ":crosstool_wrapper_driver_is_not_gcc",
    objcopy_files = ":empty",
    strip_files = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
    toolchain_identifier = "local_linux",
    toolchain_config = ":cc-compiler-local-config",
)

cc_toolchain_config(
    name = "cc-compiler-local-config",
    cpu = "local",
    compiler = "compiler",
    toolchain_identifier = "local_linux",
    host_system_name = "local",
    target_system_name = "local",
    target_libc = "local",
    abi_version = "local",
    abi_libc_version = "local",
    cxx_builtin_include_directories = [%{cxx_builtin_include_directories}],
    host_compiler_path = "%{host_compiler_path}",
    host_compiler_prefix = "%{host_compiler_prefix}",
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    dbg_compile_flags = ["-g"],
    cxx_flags = ["-std=c++17"],
    link_flags = [
        "-fuse-ld=gold",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    opt_link_flags = [],
    unfiltered_compile_flags = [
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ] + [%{unfiltered_compile_flags}],
    linker_bin_path = "%{linker_bin_path}",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    supports_start_end_lib = True,
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
  name = "crosstool_wrapper_driver_is_not_gcc",
  srcs = [":clang/bin/crosstool_wrapper_driver_is_not_gcc"],
  data = ["@local_config_rocm//rocm:all_files"],
)
