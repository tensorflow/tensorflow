load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:LICENSE"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "mnist_tflite",
    srcs = ["mnist_tflite.py"],
    strict_deps = True,
    deps = [
        ":dataset",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
