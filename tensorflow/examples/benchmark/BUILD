# Description:
# Examples of adding a benchmark to TensorFlow.

load(
    "//tensorflow/tools/test:performance.bzl",
    "tf_py_logged_benchmark",
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_test(
    name = "sample_benchmark",
    srcs = ["sample_benchmark.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_logged_benchmark(
    name = "sample_logged_benchmark",
    target = "//tensorflow/examples/benchmark:sample_benchmark",
)

filegroup(
    name = "all_files",
    srcs = glob(["**/*"]),
    visibility = ["//tensorflow:__subpackages__"],
)
