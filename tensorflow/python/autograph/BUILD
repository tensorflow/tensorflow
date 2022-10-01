load("//tensorflow:tensorflow.bzl", "py_strict_library")

package(
    licenses = ["notice"],
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
    visibility = ["//visibility:private"],
)

py_strict_library(
    name = "autograph",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/autograph/lang",
        "//tensorflow/python/autograph/operators",
        "//tensorflow/python/autograph/utils",
    ],
)
