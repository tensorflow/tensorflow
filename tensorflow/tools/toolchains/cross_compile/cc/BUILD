"""Toolchain configs for cross-compiling TensorFlow"""

load("@bazel_tools//tools/cpp:unix_cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_toolchain_suite(
    name = "cross_compile_toolchain_suite",
    toolchains = {
        "aarch64": ":linux_aarch64_toolchain",
        "k8": ":linux_x86_toolchain",
    },
)

filegroup(name = "empty")

cc_toolchain(
    name = "linux_x86_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":linux_x86_toolchain_config",
    toolchain_identifier = "linux_x86_toolchain",
)

cc_toolchain_config(
    name = "linux_x86_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/dt9",
    compile_flags = [
        "--target=x86_64-unknown-linux-gnu",
        "-fstack-protector",
        "-Wall",
        "-Wthread-safety",
        "-Wself-assign",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fcolor-diagnostics",
        "-fno-omit-frame-pointer",
        "-mavx",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "k8",
    cxx_builtin_include_directories = [
        "/dt9/",
        "/usr/lib/llvm-17/include/",
        "/usr/lib/llvm-17/lib/clang/17/include",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "linux",
    link_flags = [
        "--target=x86_64-unknown-linux-gnu",
        "-fuse-ld=lld",
        "--ld-path=/usr/lib/llvm-17/bin/ld.lld",
        "-Wl,--undefined-version",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "",
    target_system_name = "x86_64-unknown-linux-gnu",
    tool_paths = {
        "gcc": "/usr/lib/llvm-17/bin/clang",
        "ld": "/usr/lib/llvm-17/bin/ld.lld",
        "ar": "/usr/lib/llvm-17/bin/llvm-ar",
        "cpp": "/usr/lib/llvm-17/bin/clang++",
        "llvm-cov": "/usr/lib/llvm-17/bin/llvm-cov",
        "nm": "/usr/lib/llvm-17/bin/llvm-nm",
        "objdump": "/usr/lib/llvm-17/bin/llvm-objdump",
        "strip": "/usr/lib/llvm-17/bin/llvm-strip",
    },
    toolchain_identifier = "linux_x86_toolchain",
    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
        "-Wno-unused-command-line-argument",
        "-Wno-gnu-offsetof-extensions",
    ],
)

cc_toolchain(
    name = "linux_aarch64_toolchain",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":linux_aarch64_toolchain_config",
    toolchain_identifier = "linux_aarch64_toolchain",
)

cc_toolchain_config(
    name = "linux_aarch64_toolchain_config",
    abi_libc_version = "local",
    abi_version = "local",
    builtin_sysroot = "/dt10/",
    compile_flags = [
        "--target=aarch64-unknown-linux-gnu",
        "-fstack-protector",
        "-Wall",
        "-Wthread-safety",
        "-Wself-assign",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fcolor-diagnostics",
        "-fno-omit-frame-pointer",
        "-mtune=generic",
        "-march=armv8-a",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "aarch64",
    cxx_builtin_include_directories = [
        "/dt10/",
        "/usr/lib/llvm-17/include/",
        "/usr/lib/llvm-17/lib/clang/17/include",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "linux",
    link_flags = [
        "--target=aarch64-unknown-linux-gnu",
        "-fuse-ld=lld",
        "--ld-path=/usr/lib/llvm-17/bin/ld.lld",
        "-Wl,--undefined-version",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = True,
    target_libc = "",
    target_system_name = "aarch64-unknown-linux-gnu",
    tool_paths = {
        "gcc": "/usr/lib/llvm-17/bin/clang",
        "ld": "/usr/lib/llvm-17/bin/ld.lld",
        "ar": "/usr/lib/llvm-17/bin/llvm-ar",
        "cpp": "/usr/lib/llvm-17/bin/clang++",
        "llvm-cov": "/usr/lib/llvm-17/bin/llvm-cov",
        "nm": "/usr/lib/llvm-17/bin/llvm-nm",
        "objdump": "/usr/lib/llvm-17/bin/llvm-objdump",
        "strip": "/usr/lib/llvm-17/bin/llvm-strip",
    },
    toolchain_identifier = "linux_aarch64_toolchain",
    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
        "-Wno-unused-command-line-argument",
        "-Wno-gnu-offsetof-extensions",
    ],
)
