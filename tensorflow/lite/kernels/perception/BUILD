load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable", "pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "perception_ops",
    srcs = [
        "dense_image_warp.cc",
        "max_pool_with_argmax.cc",
        "max_unpooling_2d.cc",
        "perception_ops.cc",
    ],
    hdrs = [
        "perception_ops.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core/c:common",
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

cc_test(
    name = "perception_ops_test",
    size = "small",
    srcs = [
        "dense_image_warp_test.cc",
        "max_pool_with_argmax_test.cc",
        "max_unpooling_2d_test.cc",
    ],
    deps = [
        ":perception_ops",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/core:framework_stable",
        "//tensorflow/lite/kernels:test_main",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/testing:util",
        "@flatbuffers",
    ],
)

pybind_extension(
    name = "pywrap_perception_ops",
    srcs = [
        "perception_ops_wrapper.cc",
    ],
    hdrs = ["perception_ops.h"],
    additional_exported_symbols = ["PerceptionOpsRegisterer"],
    link_in_framework = True,
    deps = [
        ":perception_ops",
        "//tensorflow/lite:mutable_op_resolver",
        "//third_party/python_runtime:headers",
        "@pybind11",
    ],
)
