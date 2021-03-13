load("//tensorflow:tensorflow.bzl", "py_test")

package(
    licenses = ["notice"],  # Apache 2.0
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

py_library(
    name = "common_transformers",
    srcs = [
        "anf.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "@gast_archive//:gast",
        "@six_archive//:six",
    ],
)

py_test(
    name = "anf_test",
    srcs = ["anf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)
