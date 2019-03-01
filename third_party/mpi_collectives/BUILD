package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE.txt"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "mpi",
    srcs = select({
        "//tensorflow:macos": ["libmpi.dylib"],
        "//conditions:default": ["libmpi.so"],
    }),
    hdrs = [
        "mpi.h",
        "mpi_portable_platform",
    ],
)
