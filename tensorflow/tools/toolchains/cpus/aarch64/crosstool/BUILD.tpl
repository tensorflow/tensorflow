# This file is expanded from a template by aarch64_configure.bzl
# Update aarch64_configure.bzl#verify_build_defines when adding new variables.

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

toolchain(
    name = "toolchain-linux-aarch64",
    exec_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:aarch64",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:aarch64",
    ],
    toolchain = ":cc-compiler-local-aarch64",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local-aarch64",
        "aarch64": ":cc-compiler-local-aarch64",
    },
)

cc_toolchain(
    name = "cc-compiler-local-aarch64",
    all_files = "%{compiler_deps}",
    compiler_files = "%{compiler_deps}",
    ar_files = "%{compiler_deps}",
    as_files = "%{compiler_deps}",
    dwp_files = ":empty",
    linker_files = "%{compiler_deps}",
    objcopy_files = ":empty",
    strip_files = ":empty",
    # To support linker flags that need to go to the start of command line
    # we need the toolchain to support parameter files. Parameter files are
    # last on the command line and contain all shared libraries to link, so all
    # regular options will be left of them.
    supports_param_files = 1,
    toolchain_identifier = "local_linux_aarch64",
    toolchain_config = ":cc-compiler-local-aarch64-config",
)

cc_toolchain_config(
    name = "cc-compiler-local-aarch64-config",
    cpu = "local",
    builtin_include_directories = [%{cxx_builtin_include_directories}],
    extra_no_canonical_prefixes_flags = [%{extra_no_canonical_prefixes_flags}],
    host_compiler_path = "%{host_compiler_path}",
    host_compiler_prefix = "%{host_compiler_prefix}",
    host_compiler_warnings = [%{host_compiler_warnings}],
    host_unfiltered_compile_flags = [%{unfiltered_compile_flags}],
    linker_bin_path = "%{linker_bin_path}",
    builtin_sysroot = "%{builtin_sysroot}",
    compiler = "%{compiler}",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "aarch64_gcc_pieces",
    srcs = glob([
        "usr/aarch64-linux-gnu/**",
        "usr/libexec/**",
        "usr/lib/gcc/aarch64-unknown-linux-gnu/**",
        "usr/include/**",
    ]),
)
