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
    name = "autograph",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/autograph/lang",
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/utils",
    ],
)
