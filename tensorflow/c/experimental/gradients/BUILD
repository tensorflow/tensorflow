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
        "//tensorflow/c/eager:gradients",
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
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:gradients",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/c/experimental/ops:nn_ops",
        "//tensorflow/core/lib/llvm_rtti",
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
        "//tensorflow/c/eager:abstract_operation",
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:c_api_unified_internal",
        "//tensorflow/c/eager:gradients",
        "//tensorflow/c/experimental/ops:array_ops",
        "//tensorflow/c/experimental/ops:math_ops",
        "//tensorflow/c/experimental/ops:nn_ops",
        "//tensorflow/core/lib/llvm_rtti",
    ],
)