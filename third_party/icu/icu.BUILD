load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
])

cc_library(
    name = "headers",
    hdrs = glob(["source/common/unicode/*.h"]),
    includes = [
        "source/common",
    ],
    deps = [
    ],
)

cc_library(
    name = "common",
    hdrs = glob(["source/common/unicode/*.h"]),
    includes = [
        "source/common",
    ],
    deps = [
        ":icuuc",
    ],
)

cc_library(
    name = "icuuc",
    srcs = glob(
        [
            "source/common/**/*.c",
            "source/common/**/*.cpp",
            "source/stubdata/**/*.cpp",
        ],
    ),
    hdrs = glob([
        "source/common/*.h",
        "source/stubdata/**/*.h",
    ]),
    copts = [
        "-DU_COMMON_IMPLEMENTATION",
        "-DU_HAVE_STD_ATOMICS",  # TODO(gunan): Remove when TF is on ICU 64+.
    ] + select({
        ":android": [
            "-fdata-sections",
            "-DU_HAVE_NL_LANGINFO_CODESET=0",
            "-Wno-deprecated-declarations",
        ],
        ":apple": [
            "-Wno-shorten-64-to-32",
            "-Wno-unused-variable",
        ],
        ":windows": [
            "/utf-8",
            "/DLOCALE_ALLOW_NEUTRAL_NAMES=0",
        ],
        "//conditions:default": [],
    }),
    includes = [
        "source/stubdata",
    ],
    tags = ["requires-rtti"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":headers",
    ],
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "apple",
    values = {"cpu": "darwin"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

# This target is created to support tensorflow_text as a part of the tensorflow
cc_library(
    name = "nfkc",
    deps = [
        ":common",
    ],
)
