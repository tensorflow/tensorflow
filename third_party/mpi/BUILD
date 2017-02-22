
licenses(["notice"])  # Apache 2.0

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
        "//conditions:default": ["libmpi.so"],
    }),
    hdrs = ["mpi.h", "mpi_portable_platform.h"],
)
