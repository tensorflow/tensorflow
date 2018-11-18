licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "py_test")

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

py_library(
    name = "impl",
    srcs = [
        "api.py",
        "conversion.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/converters",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/pyct/static_analysis",
        "//tensorflow/python/autograph/utils",
        "//third_party/py/numpy",
        "@gast_archive//:gast",
        "@six_archive//:six",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/utils",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "conversion_test",
    srcs = ["conversion_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_windows"],
    deps = [
        ":impl",
        "//tensorflow/python:client_testlib",
        "@gast_archive//:gast",
    ],
)
