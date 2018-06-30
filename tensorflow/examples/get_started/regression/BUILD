licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_test(
    name = "test",
    size = "medium",
    srcs = [
        "custom_regression.py",
        "dnn_regression.py",
        "imports85.py",
        "linear_regression.py",
        "linear_regression_categorical.py",
        "test.py",
    ],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
