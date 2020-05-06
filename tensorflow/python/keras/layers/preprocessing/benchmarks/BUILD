# Benchmarks for Keras preprocessing layers.
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_py_test(
    name = "categorical_encoding_benchmark",
    srcs = ["categorical_encoding_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/keras/layers/preprocessing:categorical_encoding",
    ],
)

tf_py_test(
    name = "index_lookup_adapt_benchmark",
    srcs = ["index_lookup_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "normalization_adapt_benchmark",
    srcs = ["normalization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/keras/layers/preprocessing:normalization",
    ],
)
