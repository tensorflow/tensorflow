# Example TensorFlow models using factorization ops.

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_py_test")

tf_py_test(
    name = "mnist",
    size = "medium",
    srcs = [
        "mnist.py",
    ],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist",
        "//tensorflow/examples/tutorials/mnist:input_data",
    ],
    tags = ["notsan"],
)
