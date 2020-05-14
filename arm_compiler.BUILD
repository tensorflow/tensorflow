package(default_visibility = ["//visibility:public"])

filegroup(
    name = "gcc",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-ld",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "bin/arm-rpi-linux-gnueabihf-as",
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "arm-linux-gnueabihf/**",
        "libexec/**",
        "lib/gcc/arm-linux-gnueabihf/**",
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
