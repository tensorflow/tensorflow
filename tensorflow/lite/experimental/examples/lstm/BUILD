licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "tflite_lstm",
    srcs = ["tflite_lstm.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:framework",
        "@six_archive//:six",
    ],
)

py_library(
    name = "tflite_rnn",
    srcs = ["tflite_rnn.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:framework",
        "@six_archive//:six",
    ],
)

py_test(
    name = "unidirectional_sequence_lstm_test",
    size = "large",
    srcs = ["unidirectional_sequence_lstm_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":tflite_lstm",
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:input_data",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python/tools:optimize_for_inference",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "unidirectional_sequence_rnn_test",
    size = "large",
    srcs = ["unidirectional_sequence_rnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":tflite_rnn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:input_data",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python/tools:optimize_for_inference",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "bidirectional_sequence_lstm_test",
    size = "large",
    srcs = ["bidirectional_sequence_lstm_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":tflite_lstm",
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:input_data",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python/tools:optimize_for_inference",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "bidirectional_sequence_rnn_test",
    size = "large",
    srcs = ["bidirectional_sequence_rnn_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_oss",
        "no_pip",
    ],
    deps = [
        ":tflite_rnn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:input_data",
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python/tools:optimize_for_inference",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)
