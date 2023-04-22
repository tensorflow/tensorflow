# Benchmarks for Keras preprocessing layers.
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

tf_py_test(
    name = "category_encoding_benchmark",
    srcs = ["category_encoding_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:category_encoding",
    ],
)

tf_py_test(
    name = "category_crossing_benchmark",
    srcs = ["category_crossing_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:category_crossing",
    ],
)

tf_py_test(
    name = "hashing_benchmark",
    srcs = ["hashing_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:hashing",
    ],
)

tf_py_test(
    name = "index_lookup_adapt_benchmark",
    srcs = ["index_lookup_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "index_lookup_forward_benchmark",
    srcs = ["index_lookup_forward_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/keras/layers/preprocessing:index_lookup",
    ],
)

tf_py_test(
    name = "normalization_adapt_benchmark",
    srcs = ["normalization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:normalization",
    ],
)

tf_py_test(
    name = "discretization_adapt_benchmark",
    srcs = ["discretization_adapt_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:discretization",
    ],
)

cuda_py_test(
    name = "image_preproc_benchmark",
    srcs = ["image_preproc_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_benchmark",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers/preprocessing:image_preprocessing",
    ],
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
    name = "category_vocab_file_dense_benchmark",
    srcs = ["category_vocab_file_dense_benchmark.py"],
    python_version = "PY3",
    tags = ["no_windows"],
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
    name = "category_vocab_file_varlen_benchmark",
    srcs = ["category_vocab_file_varlen_benchmark.py"],
    python_version = "PY3",
    tags = ["no_windows"],
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
    tags = ["no_windows"],
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
    tags = ["no_windows"],
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
    tags = ["no_windows"],
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
    tags = ["no_windows"],
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
