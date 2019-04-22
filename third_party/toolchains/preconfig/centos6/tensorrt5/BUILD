# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [":tensorrt_include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tensorrt",
    srcs = [
        "tensorrt/lib/libnvinfer.so.5",
        "tensorrt/lib/libnvinfer_plugin.so.5",
    ],
    copts = cuda_default_copts(),
    data = [
        "tensorrt/lib/libnvinfer.so.5",
        "tensorrt/lib/libnvinfer_plugin.so.5",
    ],
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
        "tensorrt/lib/libnvinfer_plugin.so.5",
    ],
    cmd = """cp -f "/usr/lib64/libnvinfer.so.5" $(location tensorrt/lib/libnvinfer.so.5) && cp -f "/usr/lib64/libnvinfer_plugin.so.5" $(location tensorrt/lib/libnvinfer_plugin.so.5) """,
)

genrule(
    name = "tensorrt_include",
    outs = [
        "tensorrt/include/NvInfer.h",
        "tensorrt/include/NvUtils.h",
        "tensorrt/include/NvInferPlugin.h",
    ],
    cmd = """cp -f "/usr/include/NvInfer.h" $(location tensorrt/include/NvInfer.h) && cp -f "/usr/include/NvUtils.h" $(location tensorrt/include/NvUtils.h) && cp -f "/usr/include/NvInferPlugin.h" $(location tensorrt/include/NvInferPlugin.h) """,
)
