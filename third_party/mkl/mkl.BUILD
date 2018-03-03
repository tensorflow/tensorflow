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
    srcs = select({
        "@org_tensorflow//tensorflow:linux_x86_64": ["lib/libmklml_intel.so"],
        "@org_tensorflow//tensorflow:darwin": ["lib/libmklml.dylib"]
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libiomp5.so",
    srcs = select({
        "@org_tensorflow//tensorflow:linux_x86_64": ["lib/libiomp5.so"],
        "@org_tensorflow//tensorflow:darwin": ["lib/libiomp5.dylib"]
    }),
    visibility = ["//visibility:public"],
)
