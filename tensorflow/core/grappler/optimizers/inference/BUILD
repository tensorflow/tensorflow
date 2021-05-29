load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

DEFAULT_VISIBILITY = [
    "//tensorflow/core/grappler/optimizers/inference:__subpackages__",
    "//learning/serving:__subpackages__",
    "//tensorflow_serving:__subpackages__",
    "//tensorflow:__subpackages__",
]

package(
    default_visibility = DEFAULT_VISIBILITY,
    licenses = ["notice"],
)

tf_proto_library(
    name = "batch_op_rewriter_proto",
    srcs = ["batch_op_rewriter.proto"],
    cc_api_version = 2,
    visibility = DEFAULT_VISIBILITY,
)

cc_library(
    name = "batch_op_rewriter",
    srcs = ["batch_op_rewriter.cc"],
    hdrs = ["batch_op_rewriter.h"],
    deps = [
        ":batch_op_rewriter_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
        "//tensorflow/tools/graph_transforms:transform_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@com_google_protobuf//:protobuf_headers",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "batch_op_rewriter_test",
    srcs = ["batch_op_rewriter_test.cc"],
    deps = [
        ":batch_op_rewriter",
        ":batch_op_rewriter_proto_cc",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:graph_optimizer",
        "//tensorflow/tools/graph_transforms:transform_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)
