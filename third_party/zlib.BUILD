package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # BSD/MIT-like license (for zlib)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "zlib",
    srcs = [
        "adler32.c",
        "compress.c",
        "crc32.c",
        "crc32.h",
        "deflate.c",
        "deflate.h",
        "gzclose.c",
        "gzguts.h",
        "gzlib.c",
        "gzread.c",
        "gzwrite.c",
        "infback.c",
        "inffast.c",
        "inffast.h",
        "inffixed.h",
        "inflate.c",
        "inflate.h",
        "inftrees.c",
        "inftrees.h",
        "trees.c",
        "trees.h",
        "uncompr.c",
        "zconf.h",
        "zutil.c",
        "zutil.h",
    ],
    hdrs = ["zlib.h"],
    copts = select({
        ":windows": [],
        ":windows_msvc": [],
        "//conditions:default": [
            "-Wno-shift-negative-value",
            "-Wno-implicit-function-declaration",
        ],
    }),
    includes = ["."],
)
