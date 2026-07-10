load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])  # MIT

exports_files(["COPYING"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "windows_clang",
    constraint_values = [
        "@platforms//os:windows",
    ],
    values = {
        "compiler": "clang",
    },
)

cc_library(
    name = "farmhash",
    srcs = ["src/farmhash.cc"],
    hdrs = ["src/farmhash.h"],
    # Disable __builtin_expect support on Windows
    copts = select({
        ":windows_clang": ["-DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
    }),
    includes = ["src/."],
    visibility = ["//visibility:public"],
)
