# Placeholder: load py_proto_library
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")

DEFAULT_VISIBILITY = [
    "//tensorflow/core/grappler/optimizers/inference:__subpackages__",
    "//learning/serving:__subpackages__",
    "//tensorflow_serving:__subpackages__",
    "//tensorflow:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = DEFAULT_VISIBILITY,
    licenses = ["notice"],
)

# Expand the DEFAULT_VISIBILITY so that we can replace with public visibility with copybara.
tf_proto_library(
    name = "batch_op_rewriter_proto",
    srcs = ["batch_op_rewriter.proto"],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "batch_op_rewriter_proto_py_pb2",
#     visibility = ["//visibility:public"],
#     deps = [":batch_op_rewriter_proto"],
# )
# copybara:uncomment_end

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
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/optimizers:graph_optimizer",
        "//tensorflow/core/platform:errors",
        "//tensorflow/tools/graph_transforms:transform_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
    ],
)
