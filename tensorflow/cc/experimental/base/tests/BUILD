# Tests for the C++ header-only base types.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_test.cc",
    ],
    deps = [
        "//tensorflow/c:tf_datatype",
        "//tensorflow/cc/experimental/base/public:status",
        "//tensorflow/cc/experimental/base/public:tensor",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
