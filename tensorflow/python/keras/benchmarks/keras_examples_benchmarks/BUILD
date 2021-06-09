# Description:
#   Implementation of Keras benchmarks.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

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

COMMON_TAGS = [
    "no_pip",  # b/161253163
    "no_windows",  # b/160628318
]

cuda_py_test(
    name = "bidirectional_lstm_benchmark_test",
    srcs = ["bidirectional_lstm_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

cuda_py_test(
    name = "text_classification_transformer_benchmark_test",
    srcs = ["text_classification_transformer_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

cuda_py_test(
    name = "antirectifier_benchmark_test",
    srcs = ["antirectifier_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

cuda_py_test(
    name = "mnist_conv_benchmark_test",
    srcs = ["mnist_conv_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "mnist_hierarchical_rnn_benchmark_test",
    srcs = ["mnist_hierarchical_rnn_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

cuda_py_test(
    name = "mnist_irnn_benchmark_test",
    srcs = ["mnist_irnn_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

cuda_py_test(
    name = "reuters_mlp_benchmark_test",
    srcs = ["reuters_mlp_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "cifar10_cnn_benchmark_test",
    srcs = ["cifar10_cnn_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)

cuda_py_test(
    name = "mnist_conv_custom_training_benchmark_test",
    srcs = ["mnist_conv_custom_training_benchmark_test.py"],
    python_version = "PY3",
    tags = COMMON_TAGS,
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/keras/benchmarks:benchmark_util",
        "//tensorflow/python/keras/benchmarks:distribution_util",
        "//tensorflow/python/keras/benchmarks:profiler_lib",
    ],
)
