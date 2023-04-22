load("//tensorflow:tensorflow.bzl", "py_test")

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
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "testing",
    srcs = [
        "basic_definitions.py",
        "decorators.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "codegen",
    srcs = [
        "codegen.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/autograph/pyct",
        "//tensorflow/python/autograph/utils",
        "@gast_archive//:gast",
    ],
)

py_test(
    name = "codegen_test",
    size = "large",
    srcs = ["codegen_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "no_windows",
        "nomsan",
        "notap",
    ],
    deps = [
        ":codegen",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/autograph/pyct",
        "@gast_archive//:gast",
    ],
)
