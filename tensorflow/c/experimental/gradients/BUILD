load(
    "//tensorflow:tensorflow.bzl",
    "if_libtpu",
    "tf_cuda_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# Library of gradient functions.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
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
        "//tensorflow/c/eager:gradients_internal",
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
    visibility = ["//visibility:private"],
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
        "//tensorflow/c/eager:abstract_context",
        "//tensorflow/c/eager:gradients_internal",
    ],
)

tf_cuda_cc_test(
    name = "custom_gradient_test",
    size = "small",
    srcs = [
        "custom_gradient_test.cc",
    ],
    args = ["--heap_check="],  # TODO(b/174752220): Remove
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
    visibility = ["//tensorflow/python:__pkg__"],
)

cc_library(
    name = "grad_test_helper",
    testonly = True,
    srcs = ["grad_test_helper.cc"],
    hdrs = ["grad_test_helper.h"],
    visibility = ["//visibility:private"],
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
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":nn_grad",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:nn_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:tensor_float_32_utils",
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
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":grad_test_helper",
        ":math_grad",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:tensor_float_32_utils",
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
    args = ["--heap_check="],  # TODO(b/174752220): Remove
    tags = tf_cuda_tests_tags() + ["no_cuda_asan"],  # b/173654156,
    deps = [
        ":array_grad",
        ":grad_test_helper",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/eager:c_api_test_util",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:unified_api_testutil",
        "//tensorflow/c/experimental/gradients/tape:tape_context",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:tensor_float_32_utils",
    ] + if_libtpu(
        if_false = ["//tensorflow/compiler/mlir/tensorflow/c:mlir_c_api_registration"],
        if_true = [],
    ),
)
