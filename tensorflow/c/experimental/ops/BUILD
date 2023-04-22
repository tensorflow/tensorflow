load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Experimental ops. These will eventually be replaced by machine-generated versions.
package(
    licenses = ["notice"],
)

cc_library(
    name = "array_ops",
    srcs = [
        "array_ops.cc",
    ],
    hdrs = [
        "array_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "math_ops",
    srcs = [
        "math_ops.cc",
    ],
    hdrs = [
        "math_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":array_ops",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "nn_ops",
    srcs = [
        "nn_ops.cc",
    ],
    hdrs = [
        "nn_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "resource_variable_ops",
    srcs = [
        "resource_variable_ops.cc",
    ],
    hdrs = [
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:tracing_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
    ],
)

cc_library(
    name = "ops",
    hdrs = [
        "array_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":array_ops",
        ":math_ops",
        ":nn_ops",
        ":resource_variable_ops",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_ops.h",
        "math_ops.h",
        "nn_ops.h",
        "resource_variable_ops.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__subpackages__",
    ],
)
