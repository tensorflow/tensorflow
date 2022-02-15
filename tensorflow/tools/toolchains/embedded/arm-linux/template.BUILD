load(":cc_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

licenses(["restricted"])  # GPLv3

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "aarch64": ":cc-compiler-aarch64",
        "armhf": ":cc-compiler-armhf",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "aarch64_toolchain_all_files",
    srcs = [
        "@aarch64_linux_toolchain//:compiler_pieces",
    ],
)

filegroup(
    name = "armhf_toolchain_all_files",
    srcs = [
        "@armhf_linux_toolchain//:compiler_pieces",
    ],
)

cc_toolchain_config(
    name = "aarch64_toolchain_config",
    cpu = "aarch64",
)

cc_toolchain_config(
    name = "armhf_toolchain_config",
    cpu = "armhf",
)

cc_toolchain(
    name = "cc-compiler-aarch64",
    all_files = ":aarch64_toolchain_all_files",
    compiler_files = ":aarch64_toolchain_all_files",
    dwp_files = ":empty",
    linker_files = ":aarch64_toolchain_all_files",
    objcopy_files = "aarch64_toolchain_all_files",
    strip_files = "aarch64_toolchain_all_files",
    supports_param_files = 1,
    toolchain_config = ":aarch64_toolchain_config",
)

cc_toolchain(
    name = "cc-compiler-armhf",
    all_files = ":armhf_toolchain_all_files",
    compiler_files = ":armhf_toolchain_all_files",
    dwp_files = ":empty",
    linker_files = ":armhf_toolchain_all_files",
    objcopy_files = "armhf_toolchain_all_files",
    strip_files = "armhf_toolchain_all_files",
    supports_param_files = 1,
    toolchain_config = ":armhf_toolchain_config",
)
