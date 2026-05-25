load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")
load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

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
        ":mnist_tflite_lib",
    ],
)

py_library(
    name = "mnist_tflite_lib",
    srcs = ["mnist_tflite.py"],
    strict_deps = True,
    deps = [
        ":dataset",
        "//tensorflow:tensorflow_py",
        "//tensorflow/lite/python:lite",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "mnist_tflite_test",
    srcs = ["mnist_tflite_test.py"],
    strict_deps = True,
    deps = [
        ":mnist_tflite_lib",
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
