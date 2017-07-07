licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE"])

cc_library(
    name = "mkl_headers",
    srcs = glob(["include/*"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libmklml_intel.so",
    srcs = ["lib/libmklml_intel.so"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libiomp5.so",
    srcs = ["lib/libiomp5.so"],
    visibility = ["//visibility:public"],
)
