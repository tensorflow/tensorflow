load(":tests.bzl", "hexagon_op_tests")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "hexagon_delegate_op_model",
    testonly = 1,
    hdrs = ["hexagon_delegate_op_model.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/delegates/hexagon:hexagon_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/schema:schema_fbs",
        "@com_google_googletest//:gtest",
    ],
)

hexagon_op_tests(
    srcs = [
        "activations_test.cc",
        "arg_min_max_test.cc",
        "concat_test.cc",
        "depthwise_conv_test.cc",
        "neg_test.cc",
        "pad_test.cc",
        "pool_test.cc",
        "reduce_test.cc",
        "resize_bilinear_test.cc",
        "space_to_depth_test.cc",
        "split_test.cc",
        "transpose_conv_test.cc",
        "transpose_test.cc",
    ],
    deps = [
        ":hexagon_delegate_op_model",
        "@com_google_googletest//:gtest_main",
    ],
)
