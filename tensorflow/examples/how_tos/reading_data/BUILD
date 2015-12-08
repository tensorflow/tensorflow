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
        "//tensorflow/examples/tutorials/mnist:input_data",
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

py_binary(
    name = "fully_connected_preloaded",
    srcs = [
        "fully_connected_preloaded.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist",
        "//tensorflow/examples/tutorials/mnist:input_data",
    ],
)

py_binary(
    name = "fully_connected_preloaded_var",
    srcs = [
        "fully_connected_preloaded_var.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist",
        "//tensorflow/examples/tutorials/mnist:input_data",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
