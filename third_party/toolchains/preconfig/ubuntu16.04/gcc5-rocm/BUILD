# This file is expanded from a template by cuda_configure.bzl
# Update cuda_configure.bzl#verify_build_defines when adding new variables.

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

toolchain(
    name = "toolchain-linux-x86_64",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
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
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fno-omit-frame-pointer",
    ],
    compiler = "compiler",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "local",
    cxx_builtin_include_directories = [
        "/usr/include/c++/5",
        "/usr/include/x86_64-linux-gnu/c++/5",
        "/usr/include/c++/5/backward",
        "/usr/lib/gcc/x86_64-linux-gnu/5/include",
        "/usr/local/include",
        "/usr/lib/gcc/x86_64-linux-gnu/5/include-fixed",
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        "/opt/rocm/include",
        "/opt/rocm/hsa/include",
        "/opt/rocm/include/hip",
        "/opt/rocm/include/hip/hcc_detail",
        "/opt/rocm/hip/include",
        "/opt/rocm/llvm/lib/clang/8.0/include",
        "/opt/rocm/llvm/lib/clang/9.0.0/include",
        "/opt/rocm/llvm/lib/clang/10.0.0/include",
        "/opt/rocm/rocrand/include",
        "/opt/rocm/hiprand/include",
        "/opt/rocm/rocfft/include",
        "/opt/rocm/rocblas/include",
        "/opt/rocm/miopen/include",
        "/opt/rocm/rccl/include",
        "/opt/rocm/hcc/include",
        "/opt/rocm/hcc/compiler/lib/clang/7.0.0/include/",
        "/opt/rocm/hcc/lib/clang/7.0.0/include",
        "/opt/rocm/hcc/compiler/lib/clang/8.0.0/include/",
        "/opt/rocm/hcc/lib/clang/8.0.0/include",
        "/opt/rocm/hcc/compiler/lib/clang/9.0.0/include/",
        "/opt/rocm/hcc/lib/clang/9.0.0/include",
        "/opt/rocm/hcc/compiler/lib/clang/10.0.0/include/",
        "/opt/rocm/hcc/lib/clang/10.0.0/include",
    ],
    cxx_flags = ["-std=c++11"],
    dbg_compile_flags = ["-g"],
    host_compiler_path = "clang/bin/crosstool_wrapper_driver_is_not_gcc",
    host_compiler_prefix = "/usr/bin",
    host_system_name = "local",
    link_flags = [
        "-fuse-ld=gold",
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
        "-pass-exit-codes",
        "-lstdc++",
        "-lm",
    ],
    link_libs = [],
    linker_bin_path = "/opt/rocm/hcc/compiler/bin",
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
        "-fno-canonical-system-headers",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ] + [
        "-DTENSORFLOW_USE_ROCM=1",
        "-D__HIP_PLATFORM_HCC__",
        "-DEIGEN_USE_HIP",
    ],
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "crosstool_wrapper_driver_is_not_gcc",
    srcs = ["clang/bin/crosstool_wrapper_driver_is_not_gcc"],
)
