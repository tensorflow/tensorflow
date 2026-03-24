# Structured Tensors

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
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
    strict_deps = True,
    tags = ["nofixdeps"],
    deps = [
        ":structured_ops",
    ],
)

py_library(
    name = "structured_ops",
    srcs = ["structured_ops.py"],
    strict_deps = True,
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
    ],
)

py_library(
    name = "structured_tensor",
    srcs = ["structured_tensor.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "structured_tensor_dynamic",
    srcs = [
        "structured_tensor_dynamic.py",
    ],
    strict_deps = True,
    deps = [
        ":structured_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
    ],
)

py_library(
    name = "structured_array_ops",
    srcs = [
        "structured_array_ops.py",
    ],
    strict_deps = True,
    deps = [
        ":structured_tensor",
        "//tensorflow/core/config:flags_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
    ],
)

py_test(
    name = "structured_tensor_test",
    srcs = ["structured_tensor_test.py"],
    strict_deps = True,
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
        ":structured_tensor_dynamic",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:dispatch",
    ],
)

py_test(
    name = "structured_tensor_spec_test",
    srcs = ["structured_tensor_spec_test.py"],
    strict_deps = True,
    deps = [
        ":structured_tensor",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//tensorflow/python/platform:test",
    ],
)

py_test(
    name = "structured_array_ops_test",
    srcs = ["structured_array_ops_test.py"],
    strict_deps = True,
    deps = [
        ":structured_array_ops",
        ":structured_tensor",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops/ragged:dynamic_ragged_shape",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:row_partition",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
    ],
)

py_test(
    name = "structured_tensor_slice_test",
    srcs = ["structured_tensor_slice_test.py"],
    strict_deps = True,
    deps = [
        ":structured_tensor",
        "@absl_py//absl/testing:parameterized",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
    ],
)
