# NVIDIA TensorRT
# A high-performance deep learning inference optimizer and runtime.

licenses(["notice"])

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

cc_library(
    name = "tensorrt_headers",
    hdrs = [
        "tensorrt/include/tensorrt_config.h",
        ":tensorrt_include",
    ],
    include_prefix = "third_party/tensorrt",
    strip_include_prefix = "tensorrt/include",
)

cc_library(
    name = "tensorrt",
    srcs = [":tensorrt_lib"],
    copts = cuda_default_copts(),
    data = [":tensorrt_lib"],
    linkstatic = 1,
    deps = [
        ":tensorrt_headers",
        "@local_config_cuda//cuda",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    deps = [
        "@bazel_skylib//lib:selects",
    ],
)

genrule(
    name = "tensorrt_lib",
    outs = [
        "tensorrt/lib/libnvinfer.so.6",
        "tensorrt/lib/libnvinfer_plugin.so.6",
    ],
    cmd = """cp -f "/usr/lib/x86_64-linux-gnu/libnvinfer.so.6" "$(location tensorrt/lib/libnvinfer.so.6)" && \
cp -f "/usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.6" "$(location tensorrt/lib/libnvinfer_plugin.so.6)" """,
)

genrule(
    name = "tensorrt_include",
    outs = [
        "tensorrt/include/NvInfer.h",
        "tensorrt/include/NvUtils.h",
        "tensorrt/include/NvInferPlugin.h",
        "tensorrt/include/NvInferVersion.h",
        "tensorrt/include/NvInferRuntime.h",
        "tensorrt/include/NvInferRuntimeCommon.h",
        "tensorrt/include/NvInferPluginUtils.h",
    ],
    cmd = """cp -f "/usr/include/x86_64-linux-gnu/NvInfer.h" "$(location tensorrt/include/NvInfer.h)" && \
cp -f "/usr/include/x86_64-linux-gnu/NvUtils.h" "$(location tensorrt/include/NvUtils.h)" && \
cp -f "/usr/include/x86_64-linux-gnu/NvInferPlugin.h" "$(location tensorrt/include/NvInferPlugin.h)" && \
cp -f "/usr/include/x86_64-linux-gnu/NvInferVersion.h" "$(location tensorrt/include/NvInferVersion.h)" && \
cp -f "/usr/include/x86_64-linux-gnu/NvInferRuntime.h" "$(location tensorrt/include/NvInferRuntime.h)" && \
cp -f "/usr/include/x86_64-linux-gnu/NvInferRuntimeCommon.h" "$(location tensorrt/include/NvInferRuntimeCommon.h)" && \
cp -f "/usr/include/x86_64-linux-gnu/NvInferPluginUtils.h" "$(location tensorrt/include/NvInferPluginUtils.h)" """,
)
