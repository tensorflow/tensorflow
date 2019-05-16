package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

tf_cc_test(
    name = "lstm_utils_test",
    srcs = ["lstm_utils_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "identify_l2_pool_test",
    srcs = ["identify_l2_pool_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_concatenation_test",
    srcs = ["resolve_constant_concatenation_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "resolve_constant_unary_test",
    srcs = ["resolve_constant_unary_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "identify_l2_normalization_test",
    srcs = ["identify_l2_normalization_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "fuse_binary_into_following_affine_test",
    srcs = ["fuse_binary_into_following_affine_test.cc"],
    deps = [
        "//tensorflow/lite/toco:graph_transformations",
        "//tensorflow/lite/toco:model",
        "//tensorflow/lite/toco:tooling_util",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
    ],
)
