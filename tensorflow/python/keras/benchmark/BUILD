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

py_test(
    name = "keras_cpu_benchmark_test",
    size = "large",
    srcs = ["keras_cpu_benchmark_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "eager_microbenchmarks_test",
    size = "medium",
    srcs = ["eager_microbenchmarks_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python/keras",
    ],
)
