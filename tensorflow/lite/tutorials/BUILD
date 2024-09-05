load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "mnist_tflite",
    srcs = ["mnist_tflite.py"],
    python_version = "PY3",
    deps = [
        ":dataset",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "dataset",
    srcs = ["dataset.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
