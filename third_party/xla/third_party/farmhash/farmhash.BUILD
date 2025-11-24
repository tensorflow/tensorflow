licenses(["notice"])  # MIT

exports_files(["COPYING"])

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows",
    },
)

config_setting(
    name = "windows_x86_64_clang",
    values = {
        "compiler": "clang",
        "cpu": "x64_windows",
    },
)

cc_library(
    name = "farmhash",
    srcs = ["src/farmhash.cc"],
    hdrs = ["src/farmhash.h"],
    # Disable __builtin_expect support on Windows
    copts = select({
        ":windows_x86_64_clang": ["-DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        ":windows": ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default": [],
    }),
    includes = ["src/."],
    visibility = ["//visibility:public"],
)
