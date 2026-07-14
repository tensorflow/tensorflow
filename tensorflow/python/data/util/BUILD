load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "nest",
    srcs = ["nest.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/util:nest_util",
    ],
)

py_test(
    name = "nest_test",
    size = "small",
    srcs = ["nest_test.py"],
    strict_deps = True,
    deps = [
        ":nest",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "sparse",
    srcs = ["sparse.py"],
    strict_deps = True,
    deps = [
        ":nest",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:sparse_ops",
    ],
)

py_test(
    name = "sparse_test",
    size = "small",
    srcs = ["sparse_test.py"],
    strict_deps = True,
    deps = [
        ":nest",
        ":sparse",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "structure",
    srcs = ["structure.py"],
    strict_deps = True,
    deps = [
        ":nest",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:none_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:nest_util",
        "//tensorflow/python/util:tf_export",
        "@pypi//wrapt",
    ],
)

py_test(
    name = "structure_test",
    size = "small",
    srcs = ["structure_test.py"],
    strict_deps = True,
    deps = [
        ":nest",
        ":structure",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@pypi//wrapt",
    ],
)

py_library(
    name = "options",
    srcs = ["options.py"],
    strict_deps = True,
    deps = ["@absl_py//absl/logging"],
)

py_test(
    name = "options_test",
    size = "small",
    srcs = ["options_test.py"],
    strict_deps = True,
    deps = [
        ":options",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "convert",
    srcs = ["convert.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
    ],
)

py_test(
    name = "convert_test",
    size = "small",
    srcs = ["convert_test.py"],
    strict_deps = True,
    deps = [
        ":convert",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "random_seed",
    srcs = ["random_seed.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_test(
    name = "random_seed_test",
    size = "medium",
    srcs = ["random_seed_test.py"],
    strict_deps = True,
    deps = [
        ":random_seed",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "traverse",
    srcs = ["traverse.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:dtypes",
    ],
)

py_test(
    name = "traverse_test",
    size = "small",
    srcs = ["traverse_test.py"],
    strict_deps = True,
    deps = [
        ":traverse",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/experimental/ops:dataset_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
