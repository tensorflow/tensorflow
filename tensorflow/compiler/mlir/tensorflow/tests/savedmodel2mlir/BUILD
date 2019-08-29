load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(licenses = ["notice"])

tf_cc_test(
    name = "half_plus_two",
    srcs = ["half_plus_two.cc"],
    data = [
        "//tensorflow/cc/saved_model:saved_model_half_plus_two",
    ],
    deps = [
        "//tensorflow/cc/saved_model:tag_constants",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
