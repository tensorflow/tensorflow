# Description:
#   libuv is a cross-platform asynchronous I/O library.

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "uv",
    srcs = glob(["src/*.c"]),
    hdrs = [
        "include/uv.h",
    ],
    copts = [
        "-fexceptions",
        "-Wno-unused-variable",
    ],
    includes = [
        "include",
        "src",
    ],
    textual_hdrs = [
        "include/uv.h",
    ],
)
