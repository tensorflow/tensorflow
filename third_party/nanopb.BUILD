# Description:
#   Nanopb, a tiny ANSI C protobuf implementation for use on embedded devices.

licenses(["notice"])  # zlib license

exports_files(["LICENSE.txt"])

cc_library(
    name = "nanopb",
    srcs = [
        "pb_common.c",
        "pb_decode.c",
        "pb_encode.c",
    ],
    hdrs = [
        "pb.h",
        "pb_common.h",
        "pb_decode.h",
        "pb_encode.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)
