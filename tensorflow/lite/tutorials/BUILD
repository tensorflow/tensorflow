load("//tensorflow:strict.default.bzl", "py_strict_binary")
load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "mnist_tflite",
    srcs = ["mnist_tflite.py"],
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
