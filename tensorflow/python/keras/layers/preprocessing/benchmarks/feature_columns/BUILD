# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

tf_py_test(
    name = "bucketized_column_dense_benchmark",
    srcs = ["bucketized_column_dense_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_cross_hash_dense_benchmark",
    srcs = ["category_cross_hash_dense_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_hash_dense_benchmark",
    srcs = ["category_hash_dense_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_hash_varlen_benchmark",
    srcs = ["category_hash_varlen_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_vocab_list_dense_benchmark",
    srcs = ["category_vocab_list_dense_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_vocab_list_indicator_dense_benchmark",
    srcs = ["category_vocab_list_indicator_dense_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_vocab_list_indicator_varlen_benchmark",
    srcs = ["category_vocab_list_indicator_varlen_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "category_vocab_list_varlen_benchmark",
    srcs = ["category_vocab_list_varlen_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "embedding_dense_benchmark",
    srcs = ["embedding_dense_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "embedding_varlen_benchmark",
    srcs = ["embedding_varlen_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

py_library(
    name = "feature_column_benchmark",
    srcs = ["feature_column_benchmark.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

tf_py_test(
    name = "weighted_embedding_varlen_benchmark",
    srcs = ["weighted_embedding_varlen_benchmark.py"],
    python_version = "PY3",
    deps = [
        ":feature_column_benchmark",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)
