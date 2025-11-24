package(
    default_visibility = ["//visibility:public"],
    features = ["header_modules"],
)

licenses(["notice"])

config_setting(
    name = "x86_64",
    constraint_values = [
        "@platforms//cpu:x86_64",
    ],
)

cc_library(
    name = "zstdlib",
    srcs = glob([
        "common/*.c",
        "common/*.h",
        "compress/*.c",
        "compress/*.h",
        "decompress/*.c",
        "decompress/*.h",
    ]) + select({
        ":x86_64": glob(["decompress/*_amd64.S"]),
        "//conditions:default": [],
    }),
    hdrs = glob([
        "*.h",
    ]),
)

alias(
    name = "zstd",
    actual = ":zstdlib",
)
