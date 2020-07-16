# Description:
#   Implementation of Keras benchmarks.

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

# Add all benchmarks related utils here for pip testing dependencis.
py_library(
    name = "keras_benchmark_lib_pip",
    deps = [
        ":benchmark_util",
        ":distribution_util",
        "//tensorflow/python/keras/benchmarks/saved_model_benchmarks:saved_model_benchmark_util",
    ],
)

py_test(
    name = "keras_cpu_benchmark_test",
    size = "large",
    srcs = ["keras_cpu_benchmark_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # b/161253163
        "no_windows",  # b/160628318
    ],
    deps = [
        ":benchmark_util",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "eager_microbenchmarks_test",
    size = "medium",
    srcs = ["eager_microbenchmarks_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss_py38",  # b/160170347
        "no_windows",  # b/160269052
    ],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "model_components_benchmarks_test",
    srcs = ["model_components_benchmarks_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "benchmark_util",
    srcs = ["benchmark_util.py"],
    deps = [
        ":distribution_util",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "bidirectional_lstm_benchmark_test",
    size = "medium",
    srcs = ["keras_examples_benchmarks/bidirectional_lstm_benchmark_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # b/161253163
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
    srcs = ["keras_examples_benchmarks/text_classification_transformer_benchmark_test.py"],
    python_version = "PY3",
    tags = [
        "no_pip",  # b/161253163
        "no_windows",  # b/160628318
    ],
    deps = [
        ":benchmark_util",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "distribution_util",
    srcs = ["distribution_util.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)
