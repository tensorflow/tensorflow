load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_libtpu",
    "tf_cc_test",
    "tf_copts",
    "tf_cuda_cc_test",
    "tf_cuda_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

# Library of gradient functions.
package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "array_grad",
    srcs = ["array_grad.cc"],
    hdrs = [
        "array_grad.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/core/lib/llvm_rtti",
    ],
)

cc_library(
    name = "math_grad",
    srcs = ["math_grad.cc"],
    hdrs = [
        "math_grad.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/c/experimental/ops:nn_ops",
    ],
)

cc_library(
    name = "nn_grad",
    srcs = ["nn_grad.cc"],
    hdrs = [
        "nn_grad.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/c/experimental/ops:nn_ops",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gradients",
    hdrs = [
        "array_grad.h",
        "math_grad.h",
        "nn_grad.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":array_grad",
        ":math_grad",
        ":nn_grad",
        "//tensorflow/c/eager:gradients_internal",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_grad.h",
        "math_grad.h",
        "nn_grad.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "nn_grad_testutil",
    srcs = [
        "nn_grad_testutil.cc",
    ],
    hdrs = [
        "nn_grad_testutil.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/c/eager:gradients_util",
        "//tensorflow/c/eager:tape",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/c/experimental/ops:nn_ops",
        "//tensorflow/core/lib/llvm_rtti",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cuda_cc_test(
    name = "nn_grad_test",
    size = "small",
    srcs = [
        "nn_grad_test.cc",
    ],
    args = ["--heap_check=local"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_test_util",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/gradients:nn_grad",
        "//tensorflow/c/experimental/gradients:nn_grad_testutil",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops",
        "//tensorflow/cc/profiler",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/lib/llvm_rtti",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
