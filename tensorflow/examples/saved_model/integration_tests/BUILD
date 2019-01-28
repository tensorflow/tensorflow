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
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "saved_model_test",
    srcs = [
        "saved_model_test.py",
    ],
    data = [
        ":export_text_rnn_model",
        ":use_text_rnn_model",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
