# Structured Tensors

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow:structured_tensor_allowlist",
    ],
    licenses = ["notice"],
)

py_library(
    name = "structured",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    tags = ["nofixdeps"],
    deps = [
        ":structured_ops",
    ],
)

py_library(
    name = "structured_ops",
    srcs = ["structured_ops.py"],
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
    ],
)

py_library(
    name = "structured_tensor",
    srcs = [
        "structured_array_ops.py",
        "structured_tensor.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "structured_tensor_dynamic",
    srcs = [
        "structured_tensor_dynamic.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":structured_tensor",
    ],
)

py_library(
    name = "structured_array_ops",
    srcs = [
        "structured_array_ops.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":structured_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "structured_tensor_test",
    srcs = ["structured_tensor_test.py"],
    python_version = "PY3",
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
        ":structured_tensor_dynamic",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "structured_tensor_spec_test",
    srcs = ["structured_tensor_spec_test.py"],
    python_version = "PY3",
    deps = [
        ":structured_tensor",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/framework:type_utils",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "structured_array_ops_test",
    srcs = ["structured_array_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "structured_tensor_slice_test",
    srcs = ["structured_tensor_slice_test.py"],
    python_version = "PY3",
    deps = [
        ":structured_tensor",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "@absl_py//absl/testing:parameterized",
    ],
)
