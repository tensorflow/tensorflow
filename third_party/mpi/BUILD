licenses(["restricted"])

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

load("//third_party/mpi:mpi.bzl", "mpi_hdr")

cc_library(
    name = "mpi",
    srcs = select({
        "//conditions:default": ["libmpi.so"],
    }),
    hdrs = mpi_hdr(),
    visibility = ["//visibility:public"],
)
