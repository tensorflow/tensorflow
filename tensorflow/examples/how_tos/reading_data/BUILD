# Description:
# Example MNIST TensorFlow models for demonstrating data reading.

package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "convert_to_records",
    srcs = ["convert_to_records.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/learn/python/learn/datasets",
    ],
)

py_binary(
    name = "fully_connected_reader",
    srcs = [
        "fully_connected_reader.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist",
    ],
)
