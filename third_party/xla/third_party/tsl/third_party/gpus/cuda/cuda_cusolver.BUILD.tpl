licenses(["restricted"])  # NVIDIA proprietary license

exports_files([
    "version.txt",
])

filegroup(
    name = "include",
    srcs = ["include/cusolver_common.h", "include/cusolverDn.h", "include/cusolverSp.h"],
)

cc_import(
    name = "cusolver",
    hdrs = [":headers"],
    shared_library = "lib/libcusolver.so.%{version}",
    linkopts = ["-lgomp"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "cusolver_lib",
    srcs = ["lib/libcusolver.so.%{version}"],
)

cc_library(
    name = "headers",
    hdrs = [":include"],
    include_prefix = "third_party/gpus/cuda/include",
    includes = ["include"],
    strip_include_prefix = "include",
    visibility = ["@local_config_cuda//cuda:__pkg__"],
)
