licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_test(
    name = "python_sanity_test",
    size = "small",
    srcs = ["python_sanity_test.py"],
    data = [
        "python_modules.txt",
        "python_protos.txt",
        "python_protos_cc.txt",
        "//tensorflow:all_opensource_files",
    ],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/python:client_testlib",
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
)
