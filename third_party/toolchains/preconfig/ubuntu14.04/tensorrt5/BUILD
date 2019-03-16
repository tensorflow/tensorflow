# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

exports_files(["LICENSE"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [":tensorrt_include"],
    include_prefix = "",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorrt",
    srcs = ["tensorrt/lib/libnvinfer.so.5"],
    copts = cuda_default_copts(),
    data = ["tensorrt/lib/libnvinfer.so.5"],
    include_prefix = "",
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":tensorrt_headers",
        "@local_config_cuda//cuda",
    ],
)

genrule(
    name = "tensorrt_lib",
    outs = [
        "tensorrt/lib/libnvinfer.so.5",
    ],
    cmd = """cp -f "/usr/lib/x86_64-linux-gnu/libnvinfer.so.5" $(location tensorrt/lib/libnvinfer.so.5) """,
)

genrule(
    name = "tensorrt_include",
    outs = [
        "tensorrt/include/NvInfer.h",
        "tensorrt/include/NvUtils.h",
    ],
    cmd = """cp -f "/usr/include/x86_64-linux-gnu/NvInfer.h" $(location tensorrt/include/NvInfer.h) && cp -f "/usr/include/x86_64-linux-gnu/NvUtils.h" $(location tensorrt/include/NvUtils.h) """,
)
