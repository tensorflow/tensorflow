# Description:
# Benchmark for AlexNet.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "alexnet_benchmark",
    srcs = [
        "alexnet_benchmark.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
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
