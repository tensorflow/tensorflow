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
        "arithmetic_test.cc",
        "concat_test.cc",
        "conv_test.cc",
        "l2_norm_test.cc",
        "matmul_test.cc",
        "min_max_builder_test.cc",
        "mirror_pad_test.cc",
        "mul_test.cc",
        "neg_test.cc",
        "pack_test.cc",
        "pad_test.cc",
        "pool_test.cc",
        "quantize_test.cc",
        "reduce_test.cc",
        "reshape_test.cc",
        "resize_test.cc",
        "slice_test.cc",
        "softmax_test.cc",
        "space_to_depth_test.cc",
        "split_test.cc",
        "transpose_conv_test.cc",
        "transpose_test.cc",
    ],
    deps = [
        ":hexagon_delegate_op_model",
        "//tensorflow/lite/kernels:reshape_test_common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "batch_seq_config_test",
    srcs = [
        "batch_seq_config_test.cc",
    ],
    tags = [
        "no_oss",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/experimental/delegates/hexagon:hexagon_delegate",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels:test_util",
        "//tensorflow/lite/testing:util",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools/benchmark:benchmark_utils",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_googletest//:gtest",
    ],
)
