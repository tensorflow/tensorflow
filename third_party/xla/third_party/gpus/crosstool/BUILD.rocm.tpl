# This file is expanded from a template by rocm_configure.bzl

load(":cc_toolchain_config.bzl", "cc_toolchain_config")
load("@local_config_clang//:clang.bzl", "local_clang")

# Local clang configuration for non-hermetic toolchain
_LOCAL_CLANG = local_clang()

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

toolchain(
    name = "toolchain-linux-x86_64-hermetic",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":cc-compiler-hermetic",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-hermetic",
        "arm": ":cc-compiler-hermetic",
        "aarch64": ":cc-compiler-hermetic",
        "k8": ":cc-compiler-hermetic",
        "piii": ":cc-compiler-hermetic",
        "ppc": ":cc-compiler-hermetic",
    },
)

# Toolchain suite for local (non-hermetic) builds using system clang.
# Use with --crosstool_top=@local_config_rocm//crosstool:toolchain-local
cc_toolchain_suite(
    name = "toolchain-local",
    toolchains = {
        "local|compiler": ":cc-compiler-local-nonhermetic",
        "arm": ":cc-compiler-local-nonhermetic",
        "aarch64": ":cc-compiler-local-nonhermetic",
        "k8": ":cc-compiler-local-nonhermetic",
        "piii": ":cc-compiler-local-nonhermetic",
        "ppc": ":cc-compiler-local-nonhermetic",
    },
)

cc_toolchain(
    name = "cc-compiler-hermetic",
    all_files = "@local_config_rocm//rocm:all_files",
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
    toolchain_identifier = "local_linux_hermetic",
    toolchain_config = ":cc-compiler-hermetic-config",
)

cc_toolchain_config(
    name = "cc-compiler-hermetic-config",
    cpu = "local",
    compiler = "compiler",
    toolchain_identifier = "local_linux_hermetic",
    host_system_name = "local",
    target_system_name = "local",
    target_libc = "local",
    abi_version = "local",
    abi_libc_version = "local",
    cxx_builtin_include_directories = [
        "external/llvm18_linux_x86_64/lib/clang/18/include",
        "%sysroot%/usr/include",
        "%sysroot%/usr/include/c++/8",
        "%sysroot%/usr/include/x86_64-linux-gnu/c++/8",
        "%sysroot%/usr/include/x86_64-linux-gnu",
    ] + [%{cxx_builtin_include_directories}],
    host_compiler_path = "clang/bin/crosstool_wrapper_driver_is_not_gcc",
    host_compiler_prefix = "/usr/bin",
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        "-no-canonical-prefixes",
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
        "-fuse-ld=lld",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-Wl,--allow-shlib-undefined",
        "-Lexternal/sysroot_linux_x86_64_glibc_2_27/lib/x86_64-linux-gnu",
        "-Lexternal/sysroot_linux_x86_64_glibc_2_27/usr/lib/x86_64-linux-gnu",
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
    builtin_sysroot = "external/sysroot_linux_x86_64_glibc_2_27",
)

# =============================================================================
# Local (non-hermetic) toolchain using system compiler
# =============================================================================
toolchain(
    name = "toolchain-linux-x86_64-local",
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    target_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    toolchain = ":cc-compiler-local-nonhermetic",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-local-nonhermetic",
    all_files = "@local_config_rocm//rocm:all_files",
    compiler_files = ":crosstool_wrapper_driver_local",
    ar_files = ":crosstool_wrapper_driver_local",
    as_files = ":crosstool_wrapper_driver_local",
    dwp_files = ":empty",
    linker_files = ":crosstool_wrapper_driver_local",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "local_linux_nonhermetic",
    toolchain_config = ":cc-compiler-local-nonhermetic-config",
)

cc_toolchain_config(
    name = "cc-compiler-local-nonhermetic-config",
    cpu = "local",
    compiler = "compiler",
    toolchain_identifier = "local_linux_nonhermetic",
    host_system_name = "local",
    target_system_name = "local",
    target_libc = "local",
    abi_version = "local",
    abi_libc_version = "local",
    # Include directories detected from local clang + ROCm includes
    cxx_builtin_include_directories = _LOCAL_CLANG.include_directories + [%{cxx_builtin_include_directories}],
    host_compiler_path = "clang/bin/crosstool_wrapper_driver_is_not_gcc",
    host_compiler_prefix = "/usr/bin",
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
        "-no-canonical-prefixes",
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
        "-fuse-ld=lld",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-Wl,--allow-shlib-undefined",
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
    # Compiler path from local_clang_info(), sets CLANG_COMPILER_PATH env var
    clang_compiler_path = _LOCAL_CLANG.compiler_path,
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
  name = "crosstool_wrapper_driver_is_not_gcc",
  srcs = [
      ":clang/bin/crosstool_wrapper_driver_is_not_gcc",
      "@local_config_rocm//rocm:toolchain_data",
      "@llvm18_linux_x86_64//:clang",
      "@llvm18_linux_x86_64//:includes",
      "@llvm18_linux_x86_64//:ld",
      "@llvm18_linux_x86_64//:rt_asan",
      "@llvm18_linux_x86_64//:rt_tsan",
      "@sysroot_linux_x86_64//:sysroot",
  ],
)

# Local toolchain uses the same wrapper but CLANG_COMPILER_PATH env var
# overrides the default hermetic clang path.
filegroup(
  name = "crosstool_wrapper_driver_local",
  srcs = [
      ":clang/bin/crosstool_wrapper_driver_is_not_gcc",
      "@local_config_rocm//rocm:toolchain_data",
  ],
)
