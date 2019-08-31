# Example TensorFlow models using factorization ops.

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

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
