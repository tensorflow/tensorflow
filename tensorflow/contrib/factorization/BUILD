# Description:
# Contains ops for factorization of data, including matrix factorization and
# clustering.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

py_library(
    name = "factorization_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    data = [
        ":python/ops/_clustering_ops.so",
        ":python/ops/_factorization_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_clustering_ops",
        ":gen_factorization_ops",
    ],
)

# Ops
tf_custom_op_library(
    name = "python/ops/_clustering_ops.so",
    srcs = [
        "ops/clustering_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/factorization/kernels:clustering_ops",
    ],
)

tf_custom_op_library(
    name = "python/ops/_factorization_ops.so",
    srcs = [
        "ops/factorization_ops.cc",
    ],
    deps = [
        "//tensorflow/contrib/factorization/kernels:wals_solver_ops",
    ],
)

tf_gen_op_libs([
    "clustering_ops",
    "factorization_ops",
])

cc_library(
    name = "all_ops",
    deps = [
        ":clustering_ops_op_lib",
        ":factorization_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_clustering_ops",
    out = "python/ops/gen_clustering_ops.py",
    deps = [
        ":clustering_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_factorization_ops",
    out = "python/ops/gen_factorization_ops.py",
    deps = [
        ":factorization_ops_op_lib",
    ],
)

# Ops tests
tf_py_test(
    name = "kmeans_test",
    srcs = [
        "python/ops/kmeans_test.py",
    ],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual,",  # Flaky: b/29994267
        "notsan",
    ],
)

tf_py_test(
    name = "gmm_test",
    srcs = [
        "python/ops/gmm_test.py",
    ],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "gmm_ops_test",
    srcs = [
        "python/ops/gmm_ops_test.py",
    ],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "factorization_ops_test",
    srcs = ["python/ops/factorization_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

# Kernel tests
tf_py_test(
    name = "wals_solver_ops_test",
    srcs = ["python/kernel_tests/wals_solver_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "clustering_ops_test",
    srcs = ["python/kernel_tests/clustering_ops_test.py"],
    additional_deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

# All files
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
