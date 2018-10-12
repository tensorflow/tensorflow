licenses(["restricted"])

load("//third_party/mpi:mpi.bzl", "mpi_hdr")
load("//third_party/mpi:mpi.bzl", "if_mpi")

cc_library(
    name = "mpi",
    srcs = if_mpi([
        "libmpi.so",
    ]),
    hdrs = if_mpi(mpi_hdr()),
    visibility = ["//visibility:public"],
)
