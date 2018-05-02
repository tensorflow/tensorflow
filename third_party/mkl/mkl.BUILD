licenses(["notice"])  # 3-Clause BSD

exports_files(["license.txt"])

filegroup(
    name = "LICENSE",
    srcs = [
        "license.txt",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mkl_headers",
    srcs = glob(["include/*"]),
    visibility = ["//visibility:public"],
)

load("@org_tensorflow//tensorflow:tensorflow.bzl",
     "if_darwin",
     "if_linux_x86_64",
     "if_windows")

filegroup(
    name = "libmklml",
    srcs = if_darwin(["lib/libmklml.dylib"])
         + if_linux_x86_64(["lib/libmklml_intel.so"])
         + if_windows(["lib/mklml.lib"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "libiomp5",
    srcs = if_darwin(["lib/libiomp5.dylib"])
         + if_linux_x86_64(["lib/libiomp5.so"])
         + if_windows(["lib/libiomp5md.lib"]),
    visibility = ["//visibility:public"],
)
