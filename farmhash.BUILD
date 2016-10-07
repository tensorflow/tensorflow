licenses(["notice"])  # MIT

config_setting(
    name = "windows",
    values = {
        "cpu": "x64_windows_msvc",
    },
)


cc_library(
    name = "farmhash",
    srcs = ["farmhash.cc"],
    hdrs = ["farmhash.h"],
    # Disable __builtin_expect support on Windows
    copts = select({
        ":windows" : ["/DFARMHASH_OPTIONAL_BUILTIN_EXPECT"],
        "//conditions:default" : [],
    }),
    includes = ["."],
    visibility = ["//visibility:public"],
)
