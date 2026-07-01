load("@rules_cc//cc:cc_library.bzl", "cc_library")

licenses(["notice"])  # MIT

exports_files(["COPYING"])

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
)

config_setting(
    name = "windows_x86_64_clang",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang-cl",
    ],
)

config_setting(
    name = "windows_arm64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:arm64",
    ],
)

cc_library(
    name = "farmhash",
    srcs = ["src/farmhash.cc"],
    hdrs = ["src/farmhash.h"],
    # Disable __builtin_expect support on Windows
    copts = select({
        ":windows_x86_64_clang": ["-DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows_arm64": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
    }),
    includes = ["src/."],
    visibility = ["//visibility:public"],
)
