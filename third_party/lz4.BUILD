package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

licenses(["notice"])

cc_library(
    name = "lz4",
    srcs = glob([
        "lib/*.c",
        "lib/*.h"
    ]),
    hdrs = ["lib/lz4frame.h"],
    copts = [
    ],
    includes = ["lib/"],
    linkopts = select({
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [],
    }),
)
