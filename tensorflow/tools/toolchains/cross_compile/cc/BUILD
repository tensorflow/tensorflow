"""Toolchain configs for cross-compiling TensorFlow"""

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])

cc_toolchain_suite(
    name = "cross_compile_toolchain_suite",
    toolchains = {
        "aarch64": ":linux_aarch64_toolchain",
        "k8": ":linux_x86_toolchain",
        "darwin": ":macos_x86_toolchain",
    },
)

filegroup(name = "empty")

# We define a wraper ("cc_wrapper.sh") around the compiler to replace all paths
# in the binary (bazel-out/.../path/to/original/library.so) by the paths
# relative to the binary. Without it, we run into "Library not loaded" error
# when trying run cross-compiled tests, see b/300002682.
filegroup(
    name = "cc_wrapper_and_macos_sysroot",
    srcs = ["cc_wrapper.sh"] + glob(["MacOSX.sdk/**"]),
)

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

cc_toolchain(
    name = "macos_x86_toolchain",
    all_files = ":cc_wrapper_and_macos_sysroot",
    compiler_files = ":cc_wrapper_and_macos_sysroot",
    dwp_files = ":empty",
    linker_files = ":cc_wrapper_and_macos_sysroot",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":macos_x86_toolchain_config",
    toolchain_identifier = "macos_x86_toolchain",
)

cc_toolchain_config(
    name = "macos_x86_toolchain_config",
    abi_libc_version = "darwin_x86_64",
    abi_version = "darwin_x86_64",
    builtin_sysroot = "tensorflow/tools/toolchains/cross_compile/cc/MacOSX.sdk",
    compile_flags = [
        "--target=x86_64-apple-darwin",
        "-fstack-protector",
        "-Wall",
        "-Wthread-safety",
        "-Wself-assign",
        "-Wunused-but-set-parameter",
        "-Wno-free-nonheap-object",
        "-fcolor-diagnostics",
        "-fno-omit-frame-pointer",
        "-DOS_MACOSX",
        "-DGRPC_BAZEL_BUILD",
        "-stdlib=libc++",
        "-mavx",
        # Target Catalina as the minimum supported OS
        "-mmacos-version-min=10.15",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "darwin",
    cxx_builtin_include_directories = [
        "%sysroot%/usr/include",
        "/usr/lib/llvm-17/include/",
        "/usr/lib/llvm-17/lib/clang/17/include",
        "%sysroot%/System/Library/Frameworks/Security.framework/Headers",
        "%sysroot%/System/Library/Frameworks/CoreFoundation.framework/Headers",
        "%sysroot%/System/Library/Frameworks/SystemConfiguration.framework/Headers",
    ],
    dbg_compile_flags = ["-g"],
    host_system_name = "linux",
    link_flags = [
        "--target=x86_64-apple-darwin",
        "-lSystem",
        "-fuse-ld=lld",
        "--ld-path=/usr/lib/llvm-17/bin/ld64.lld",
        "-headerpad_max_install_names",
        "-Wl,-undefined,dynamic_lookup",
        # Target Catalina as the minimum supported OS
        "-Wl,-platform_version,macos,10.15.0,10.15",
    ],
    link_libs = [
        "-lc++",
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
    opt_link_flags = ["-Wl,-dead_strip"],
    supports_start_end_lib = True,
    target_libc = "macosx",
    target_system_name = "x86_64-apple-macosx10.15",
    tool_paths = {
        "gcc": "cc_wrapper.sh",
        "ld": "/usr/lib/llvm-17/bin/ld64.lld",
        "ar": "/usr/lib/llvm-17/bin/llvm-libtool-darwin",
        "cpp": "/usr/lib/llvm-17/bin/clang++",
        "llvm-cov": "/usr/lib/llvm-17/bin/llvm-cov",
        "nm": "/usr/lib/llvm-17/bin/llvm-nm",
        "objdump": "/usr/lib/llvm-17/bin/llvm-objdump",
        "strip": "/usr/lib/llvm-17/bin/llvm-strip",
    },
    toolchain_identifier = "macos_x86_toolchain",
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
