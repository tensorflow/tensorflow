licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "tfe",
    srcs = ["tfe.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":saver",
        "//tensorflow/python:framework_ops",
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
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
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
