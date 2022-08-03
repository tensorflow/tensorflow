# Tests for DTensor interfaces.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

tf_cc_test(
    name = "tensor_layout_test",
    srcs = ["tensor_layout_test.cc"],
    deps = [
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/dtensor/cc:tensor_layout",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)
