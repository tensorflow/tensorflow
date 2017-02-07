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
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "factorization_py",
    srcs = [
        "__init__.py",
        "python/ops/clustering_ops.py",
        "python/ops/factorization_ops.py",
        "python/ops/gmm.py",
        "python/ops/gmm_ops.py",
    ],
    data = [
        ":python/ops/_clustering_ops.so",
        ":python/ops/_factorization_ops.so",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":gen_clustering_ops",
        ":gen_factorization_ops",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
    deps = [
        "//tensorflow/contrib/learn",
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
        "//tensorflow/contrib/factorization/kernels:masked_matmul_ops",
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
    name = "gmm_test",
    srcs = [
        "python/ops/gmm_test.py",
    ],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "notsan",  # Flaky: b/30756419
    ],
)

tf_py_test(
    name = "gmm_ops_test",
    srcs = [
        "python/ops/gmm_ops_test.py",
    ],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "factorization_ops_test",
    srcs = ["python/ops/factorization_ops_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

# Kernel tests
tf_py_test(
    name = "wals_solver_ops_test",
    srcs = ["python/kernel_tests/wals_solver_ops_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "clustering_ops_test",
    srcs = ["python/kernel_tests/clustering_ops_test.py"],
    additional_deps = [
        ":factorization_py",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

tf_py_test(
    name = "masked_matmul_ops_test",
    srcs = ["python/kernel_tests/masked_matmul_ops_test.py"],
    additional_deps = [
        ":gen_factorization_ops",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "masked_matmul_benchmark",
    srcs = ["python/kernel_tests/masked_matmul_benchmark.py"],
    additional_deps = [
        ":gen_factorization_ops",
        ":factorization_py_CYCLIC_DEPENDENCIES_THAT_NEED_TO_GO",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:variables",
    ],
    main = "python/kernel_tests/masked_matmul_benchmark.py",
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
