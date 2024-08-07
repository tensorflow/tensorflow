load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:types",
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
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

pybind_extension(
    name = "pywrap_gradient_ops",
    srcs = [
        "gradient_ops_wrapper.cc",
    ],
    hdrs = ["gradient_ops.h"],
    additional_exported_symbols = ["GradientOpsRegisterer"],
    enable_stub_generation = True,
    link_in_framework = True,
    pytype_srcs = [
        "pywrap_gradient_ops.pyi",
    ],
    deps = [
        ":gradient_ops",
        "//tensorflow/lite:mutable_op_resolver",
        "@pybind11",
    ],
)
