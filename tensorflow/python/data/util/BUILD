load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "nest",
    srcs = ["nest.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python:util"],
)

py_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":nest",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/ops/ragged",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "sparse",
    srcs = ["sparse.py"],
    srcs_version = "PY3",
    deps = [
        ":nest",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/user_ops:ops",
    ],
)

py_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":nest",
        ":sparse",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    srcs_version = "PY3",
    deps = [
        ":nest",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/user_ops:ops",
        "@wrapt",
    ],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":nest",
        ":structure",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@wrapt",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    srcs_version = "PY3",
)

py_test(
    name = "options_test",
    size = "small",
    srcs = ["options_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":options",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

py_test(
    name = "convert_test",
    size = "small",
    srcs = ["convert_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":convert",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

py_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
    ],
)

py_test(
    name = "random_seed_test",
    size = "small",
    srcs = ["random_seed_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":random_seed",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:util",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

py_library(
    name = "traverse",
    srcs = ["traverse.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_test(
    name = "traverse_test",
    size = "small",
    srcs = ["traverse_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":traverse",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/data/experimental/kernel_tests/service:test_base",
        "//tensorflow/python/data/experimental/ops:dataset_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)
