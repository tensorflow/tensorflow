# NVIDIA TensorRT Open Source Plugins
# This package contains build targets for select TensorRT plugins included in the
# TensorRT open source repository.
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts", "cuda_library")

exports_files(["LICENSE"])

cuda_library(
    name = "plugin_common",
    srcs = [
        "plugin/common/kernels/common.cu.cc",
    ],
    hdrs = [
        "plugin/common/bboxUtils.h",
        "plugin/common/checkMacrosPlugin.h",
        "plugin/common/plugin.h",
    ],
    strip_include_prefix = "plugin/common",
    deps = [
        "//third_party/tensorrt:nv_infer_headers",
        "@local_config_tensorrt//:tensorrt",
    ],
)

cc_library(
    name = "nms_plugin_hdrs",
    hdrs = [
        "plugin/efficientNMSPlugin/efficientNMSInference.h",
        "plugin/efficientNMSPlugin/efficientNMSParameters.h",
        "plugin/efficientNMSPlugin/efficientNMSPlugin.h",
    ],
)

cuda_library(
    name = "nvinfer_plugin_nms",
    srcs = [
        "plugin/efficientNMSPlugin/efficientNMSInference.cu.cc",
        "plugin/efficientNMSPlugin/efficientNMSInference.cu.h",
        "plugin/efficientNMSPlugin/efficientNMSInference.h",
        "plugin/efficientNMSPlugin/efficientNMSParameters.h",
        "plugin/efficientNMSPlugin/efficientNMSPlugin.cpp",
        "plugin/efficientNMSPlugin/efficientNMSPlugin.h",
    ],
    hdrs = [
        "plugin/efficientNMSPlugin/efficientNMSInference.h",
        "plugin/efficientNMSPlugin/efficientNMSParameters.h",
        "plugin/efficientNMSPlugin/efficientNMSPlugin.h",
    ],
    copts = cuda_default_copts(),
    include_prefix = "third_party/tensorrt/plugin/efficientNMSPlugin",
    strip_include_prefix = "plugin/efficientNMSPlugin",
    visibility = ["//visibility:public"],
    deps = [
        ":nms_plugin_hdrs",
        ":plugin_common",
        "//third_party/tensorrt:nv_infer_headers",
        "@local_config_tensorrt//:tensorrt",
    ],
)
