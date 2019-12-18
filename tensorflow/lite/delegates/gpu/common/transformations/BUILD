package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "add_bias",
    srcs = ["add_bias.cc"],
    hdrs = ["add_bias.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_library(
    name = "fuse_add_to_conv",
    srcs = ["fuse_add_to_conv.cc"],
    hdrs = ["fuse_add_to_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "fuse_add_to_conv_test",
    srcs = ["fuse_add_to_conv_test.cc"],
    deps = [
        ":fuse_add_to_conv",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "fuse_mul_to_conv",
    srcs = ["fuse_mul_to_conv.cc"],
    hdrs = ["fuse_mul_to_conv.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common:tensor",
    ],
)

cc_test(
    name = "fuse_mul_to_conv_test",
    srcs = ["fuse_mul_to_conv_test.cc"],
    deps = [
        ":fuse_mul_to_conv",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "general_transformations",
    srcs = ["general_transformations.cc"],
    hdrs = ["general_transformations.h"],
    deps = [
        ":fuse_add_to_conv",
        ":fuse_mul_to_conv",
        ":make_fully_connected",
        ":make_padding",
        ":match_dilated_convolution",
        ":merge_padding_with",
        ":remove_noop",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
    ],
)

cc_library(
    name = "make_fully_connected",
    srcs = ["make_fully_connected.cc"],
    hdrs = ["make_fully_connected.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "make_fully_connected_test",
    srcs = ["make_fully_connected_test.cc"],
    deps = [
        ":make_fully_connected",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "make_padding",
    srcs = ["make_padding.cc"],
    hdrs = ["make_padding.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "make_padding_test",
    srcs = ["make_padding_test.cc"],
    deps = [
        ":make_padding",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "matching",
    hdrs = ["matching.h"],
    deps = ["//tensorflow/lite/delegates/gpu/common:model"],
)

cc_library(
    name = "match_dilated_convolution",
    srcs = ["match_dilated_convolution.cc"],
    hdrs = ["match_dilated_convolution.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "match_dilated_convolution_test",
    srcs = ["match_dilated_convolution_test.cc"],
    deps = [
        ":match_dilated_convolution",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "merge_padding_with",
    srcs = ["merge_padding_with.cc"],
    hdrs = ["merge_padding_with.h"],
    deps = [
        ":matching",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:any",
    ],
)

cc_test(
    name = "merge_padding_with_test",
    srcs = ["merge_padding_with_test.cc"],
    deps = [
        ":merge_padding_with",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "@com_google_absl//absl/types:any",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "remove_noop",
    srcs = ["remove_noop.cc"],
    hdrs = ["remove_noop.h"],
    deps = [
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "//tensorflow/lite/delegates/gpu/common:status",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "remove_noop_test",
    srcs = ["remove_noop_test.cc"],
    deps = [
        ":remove_noop",
        "//tensorflow/lite/delegates/gpu/common:data_type",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:operations",
        "@com_google_googletest//:gtest_main",
    ],
)
