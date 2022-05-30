load("//tensorflow:tensorflow.bzl", "py_strict_test")

package(
    default_visibility = ["//tensorflow/core/function:__subpackages__"],
)

licenses(["notice"])

py_strict_test(
    name = "side_inputs_test",
    srcs = ["side_inputs_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)
