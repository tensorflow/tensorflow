# Repository initialized in tensorflow/workspace2.bzl
load(":cc_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

# The following line is only here to make this project import into IDEs that embed
# a Bazel toolchain.
licenses(["notice"])

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "local|compiler": ":cc-compiler-local",
        "aarch64": ":cc-compiler-aarch64",
        "k8": ":cc-compiler-local",
        "piii": ":cc-compiler-local",
        "arm": ":cc-compiler-local",
        "s390x": ":cc-compiler-local",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "arm_linux_all_files",
    srcs = [
        "@arm_compiler//:compiler_pieces",
    ],
)

filegroup(
    name = "aarch64_linux_all_files",
    srcs = [
        "@aarch64_compiler//:aarch64_compiler_pieces",
    ],
)

cc_toolchain_config(
    name = "local_config",
    cpu = "local",
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local_config",
    toolchain_identifier = "local_linux",
)

cc_toolchain_config(
    name = "aarch64_config",
    cpu = "aarch64",
)

cc_toolchain(
    name = "cc-compiler-aarch64",
    all_files = ":aarch64_linux_all_files",
    compiler_files = ":aarch64_linux_all_files",
    dwp_files = ":empty",
    linker_files = ":aarch64_linux_all_files",
    objcopy_files = "aarch64_linux_all_files",
    strip_files = "aarch64_linux_all_files",
    supports_param_files = 1,
    toolchain_config = ":aarch64_config",
    toolchain_identifier = "aarch64-linux-gnu",
)
