package(default_visibility = ["//visibility:public"])

# Please note that the output of these tools is unencumbered.
licenses(["restricted"])  # NCSA, GPLv3 (e.g. gold)

filegroup(
    name = "ar",
    srcs = ["llvm/bin/llvm-ar"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "as",
    srcs = ["llvm/bin/llvm-as"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "cpp",
    srcs = ["llvm/bin/llvm-cpp"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "dwp",
    srcs = ["llvm/bin/llvm-dwp"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "gcc",
    srcs = ["llvm/bin/clang"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "gcov",
    srcs = ["llvm/bin/llvm-cov"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "ld",
    srcs = ["llvm/bin/ld.lld"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "nm",
    srcs = ["llvm/bin/llvm-nm"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "objcopy",
    srcs = ["llvm/bin/llvm-objcopy"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "objdump",
    srcs = ["llvm/bin/llvm-objdump"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "profdata",
    srcs = ["llvm/bin/llvm-profdata"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "strip",
    srcs = ["sbin/strip"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "xray",
    srcs = ["llvm/bin/llvm-xray"],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "includes",
    srcs = glob(["llvm/lib/clang/6.0.0/include/**"]),
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "libraries",
    srcs = glob([
        "lib/*.*",
        "lib/clang/6.0.0/lib/linux/*.*",
    ]),
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "compiler_files",
    srcs = [
        ":as",
        ":gcc",
        ":includes",
    ],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "linker_files",
    srcs = [
        ":ar",
        ":ld",
        ":libraries",
    ],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "all_files",
    srcs = [
        ":compiler_files",
        ":dwp",
        ":gcov",
        ":linker_files",
        ":nm",
        ":objcopy",
        ":objdump",
        ":profdata",
        ":strip",
        ":xray",
    ],
    output_licenses = ["unencumbered"],
)

filegroup(
    name = "empty",
    srcs = [],  # bazel crashes without this
    output_licenses = ["unencumbered"],
)

cc_toolchain_suite(
    name = "clang6",
    toolchains = {
        "k8|clang6": ":clang6-k8",
    },
)

cc_toolchain(
    name = "clang6-k8",
    all_files = ":all_files",
    compiler_files = ":compiler_files",
    cpu = "k8",
    dwp_files = ":dwp",
    linker_files = ":linker_files",
    objcopy_files = ":objcopy",
    output_licenses = ["unencumbered"],
    strip_files = ":strip",
    supports_param_files = 1,
)
