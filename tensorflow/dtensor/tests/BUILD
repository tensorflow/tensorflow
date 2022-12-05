# Tests for DTensor interfaces.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

tf_cc_test(
    name = "tensor_layout_test",
    srcs = ["tensor_layout_test.cc"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "layout_to_xla_sharding_test",
    srcs = ["layout_to_xla_sharding_test.cc"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:layout_to_xla_sharding",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)
