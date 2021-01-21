load("//tensorflow:tensorflow.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load(
    "//tensorflow:tensorflow.bzl",
    "if_libtpu",
    "tf_cuda_cc_test",
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
        "//tensorflow/c/eager:abstract_context",
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
    name = "not_differentiable",
    srcs = ["not_differentiable.cc"],
    hdrs = [
        "not_differentiable.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:gradients_internal",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gradients",
    hdrs = [
        "array_grad.h",
        "math_grad.h",
        "nn_grad.h",
        "not_differentiable.h",
    ],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        ":array_grad",
        ":math_grad",
        ":nn_grad",
        ":not_differentiable",
        "//tensorflow/c/eager:gradients_internal",
    ],
)

tf_cuda_cc_test(
    name = "custom_gradient_test",
    size = "small",
    srcs = [
        "custom_gradient_test.cc",
    ],
    args = ["--heap_check=local"],  # TODO(b/174752220): Remove
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags(),
    deps = [
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/ops",
        "//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "array_grad.h",
        "math_grad.h",
        "nn_grad.h",
        "not_differentiable.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
)

cc_library(
    name = "grad_test_helper",
    testonly = True,
    srcs = ["grad_test_helper.cc"],
    hdrs = ["grad_test_helper.h"],
    visibility = [
        "//tensorflow:internal",
    ],
    deps = [
        "//tensorflow/c/eager:gradient_checker",
        "//tensorflow/c/eager:gradients_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_cc_test(
    name = "nn_grad_test",
    size = "small",
    srcs = [
        "nn_grad_test.cc",
    ],
    args = ["--heap_check=local"],  # TODO(b/174752220): Remove
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":nn_grad",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:nn_ops",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_libtpu(
        if_false = ["//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration"],
        if_true = [],
    ),
)

tf_cuda_cc_test(
    name = "math_grad_test",
    size = "small",
    srcs = [
        "math_grad_test.cc",
    ],
    args = ["--heap_check=local"],  # TODO(b/174752220): Remove
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":math_grad",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_libtpu(
        if_false = ["//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration"],
        if_true = [],
    ),
)

tf_cuda_cc_test(
    name = "array_grad_test",
    size = "small",
    srcs = [
        "array_grad_test.cc",
    ],
    args = ["--heap_check=local"],  # TODO(b/174752220): Remove
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":array_grad",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/core/platform:tensor_float_32_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ] + if_libtpu(
        if_false = ["//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration"],
        if_true = [],
    ),
)
