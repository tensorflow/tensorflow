# Description:
#   Implementation of benchmarks for keras.io examples.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

# To run CPU benchmarks:
#   bazel run -c opt benchmarks_test -- --benchmarks=.

# To run GPU benchmarks:
#   bazel run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \
#     --benchmarks=.

# To run a subset of benchmarks using --benchmarks flag.
# --benchmarks: the list of benchmarks to run. The specified value is interpreted
# as a regular expression and any benchmark whose name contains a partial match
# to the regular expression is executed.
# e.g. --benchmarks=".*lstm*." will run all lstm layer related benchmarks.

py_test(
    name = "bidirectional_lstm_benchmark_test",
    size = "medium",
    srcs = ["bidirectional_lstm_benchmark_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  # b/160628318
    ],
    deps = [
        ":benchmark_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "text_classification_transformer_benchmark_test",
    size = "medium",
    srcs = ["text_classification_transformer_benchmark_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  # b/160628318
    ],
    deps = [
        ":benchmark_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "benchmark_util",
    srcs = ["benchmark_util.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)
