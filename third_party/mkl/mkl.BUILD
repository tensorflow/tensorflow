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

cc_library(
    name = "mkl_libs_linux",
    srcs = [
        "lib/libiomp5.so",
        "lib/libmklml_intel.so",
        "lib/libnuma.so"
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_darwin",
    srcs = [
        "lib/libiomp5.dylib",
        "lib/libmklml.dylib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_windows",
    srcs = [
        "lib/libiomp5md.lib",
        "lib/mklml.lib",
    ],
    visibility = ["//visibility:public"],
)
