licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_binary(
    name = "export_text_rnn_model",
    srcs = ["export_text_rnn_model.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "use_text_rnn_model",
    srcs = ["use_text_rnn_model.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "export_rnn_cell",
    srcs = ["export_rnn_cell.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "use_rnn_cell",
    srcs = ["use_rnn_cell.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "export_simple_text_embedding",
    srcs = ["export_simple_text_embedding.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "use_model_in_sequential_keras",
    srcs = ["use_model_in_sequential_keras.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "mnist_util",
    srcs = ["mnist_util.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "export_mnist_cnn",
    srcs = ["export_mnist_cnn.py"],
    deps = [
        ":mnist_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "use_mnist_cnn",
    srcs = ["use_mnist_cnn.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":mnist_util",
        ":util",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "saved_model_test",
    srcs = [
        "saved_model_test.py",
    ],
    data = [
        ":export_mnist_cnn",
        ":export_rnn_cell",
        ":export_simple_text_embedding",
        ":export_text_rnn_model",
        ":use_mnist_cnn",
        ":use_model_in_sequential_keras",
        ":use_rnn_cell",
        ":use_text_rnn_model",
    ],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "noasan",  # forge input size exceeded
        "nomsan",  # forge input size exceeded
        "notsan",  # forge input size exceeded
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
