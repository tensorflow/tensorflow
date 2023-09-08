package(default_visibility = ["//visibility:public"])

filegroup(
    name = "gcc",
    srcs = [
        "bin/aarch64-none-linux-gnu-gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "bin/aarch64-none-linux-gnu-ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "bin/aarch64-none-linux-gnu-ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "bin/aarch64-none-linux-gnu-nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "bin/aarch64-none-linux-gnu-objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "bin/aarch64-none-linux-gnu-objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "bin/aarch64-none-linux-gnu-strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "bin/aarch64-none-linux-gnu-as",
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "aarch64-none-linux-gnu/**",
        "libexec/**",
        "lib/gcc/aarch64-none-linux-gnu/**",
        "include/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
