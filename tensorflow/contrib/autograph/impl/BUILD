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
        "//tensorflow/contrib/autograph/converters",
        "//tensorflow/contrib/autograph/core",
        "//tensorflow/contrib/autograph/operators",
        "//tensorflow/contrib/autograph/pyct",
        "//tensorflow/contrib/autograph/pyct/static_analysis",
        "//tensorflow/contrib/autograph/utils",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
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
        "//tensorflow/contrib/autograph/utils",
        "//tensorflow/python:client_testlib",
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
