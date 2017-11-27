licenses(["notice"])  # 3-Clause BSD

exports_files(["license.txt"])

filegroup(
    name = "LICENSE",
    srcs = [
        "license.txt",
    ],
    visibility = ["//visibility:public"],
)

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
