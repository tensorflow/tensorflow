licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test", "cuda_py_test")

py_library(
    name = "tfe",
    srcs = ["tfe.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        ":saver",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:core",
        "//tensorflow/python/eager:custom_gradient",
        "//tensorflow/python/eager:execution_callbacks",
        "//tensorflow/python/eager:function",
    ],
)

cuda_py_test(
    name = "tfe_test",
    srcs = ["tfe_test.py"],
    additional_deps = [
        ":tfe",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/contrib/data/python/ops:dataset_ops",
        "//tensorflow/contrib/data/python/util:nest",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:errors",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        "//tensorflow/contrib/data",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "saver",
    srcs = ["saver.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:training",
    ],
)

cuda_py_test(
    name = "saver_test",
    srcs = ["saver_test.py"],
    additional_deps = [
        ":saver",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
