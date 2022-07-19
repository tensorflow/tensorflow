# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "bcast_grad_args_op",
    srcs = [
        "bcast_grad_args.cc",
    ],
    hdrs = [
        "bcast_grad_args.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:tensor_utils",
        "//tensorflow/lite/kernels/internal:types",
        "@flatbuffers",
    ],
)

cc_library(
    name = "gradient_ops",
    srcs = [
        "gradient_ops.cc",
    ],
    hdrs = [
        "gradient_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":bcast_grad_args_op",
        "//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "bcast_grad_args_op_test",
    size = "small",
    srcs = [
        "bcast_grad_args_test.cc",
    ],
    deps = [
        ":bcast_grad_args_op",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/testing:util",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_gradient_ops",
    srcs = [
        "gradient_ops_wrapper.cc",
    ],
    hdrs = ["gradient_ops.h"],
    additional_exported_symbols = ["GradientOpsRegisterer"],
    link_in_framework = True,
    deps = [
        ":gradient_ops",
        "//tensorflow/lite:mutable_op_resolver",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)
