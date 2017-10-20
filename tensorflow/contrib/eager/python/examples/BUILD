# Description:
#   Open-source examples and tutorials for TensorFlow Eager Execution.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_binary(
    name = "linear_regression",
    srcs = ["linear_regression.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "@six_archive//:six",
    ],
)

py_library(
    name = "cart_pole_helper",
    srcs = ["cart_pole_helper.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "spinn",
    srcs = ["spinn.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "cart_pole",
    srcs = ["cart_pole.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cart_pole_helper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "spinn_prep_data",
    srcs = ["spinn_prep_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "spinn_train",
    srcs = ["spinn_train.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":spinn",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/eager/python:tfe",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "linear_regression_test",
    size = "small",
    srcs = ["tests/linear_regression_test.py"],
    additional_deps = [
        ":linear_regression",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "cart_pole_helper_test",
    srcs = ["tests/cart_pole_helper_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cart_pole_helper",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

cuda_py_test(
    name = "cart_pole_test",
    size = "small",
    srcs = ["tests/cart_pole_test.py"],
    additional_deps = [
        ":cart_pole",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:training",
    ],
)

cuda_py_test(
    name = "spinn_test",
    size = "medium",
    srcs = ["tests/spinn_test.py"],
    additional_deps = [
        ":spinn",
        "//third_party/py/numpy",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
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
