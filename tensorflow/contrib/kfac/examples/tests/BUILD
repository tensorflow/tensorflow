package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "mlp_test",
    size = "large",
    srcs = ["mlp_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/kfac/examples:mlp",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "convnet_test",
    size = "large",
    srcs = ["convnet_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
        "notsan",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/kfac",
        "//tensorflow/contrib/kfac/examples:convnet",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "mnist_test",
    srcs = ["mnist_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/kfac/examples:mnist",
        "//third_party/py/numpy",
    ],
)
