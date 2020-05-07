# Description:
#   C-style (a-la printf) logging library

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "clog",
    srcs = [
        "deps/clog/src/clog.c",
    ],
    hdrs = [
        "deps/clog/include/clog.h",
    ],
    copts = select({
        ":windows": [],
        "//conditions:default": ["-Wno-unused-result"],
    }),
    linkopts = select({
        ":android": ["-llog"],
        "//conditions:default": [],
    }),
    linkstatic = True,
    strip_include_prefix = "deps/clog/include",
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)
