# Tests for the C++ header-only SavedModelAPI.
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_cc_test(
    name = "saved_model_api_test",
    srcs = [
        "saved_model_api_test.cc",
    ],
    deps = [
        "//tensorflow/cc/experimental/base/public:runtime",
        "//tensorflow/cc/experimental/base/public:runtime_builder",
        "//tensorflow/cc/experimental/base/public:status",
        "//tensorflow/cc/saved_model/experimental/public:saved_model_api",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
