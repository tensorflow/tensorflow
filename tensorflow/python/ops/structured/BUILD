# Structured Tensors

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = [
        "//learning/tfx/autotfx:__subpackages__",
        "//research/graph/convolutions/model/autotfx:__subpackages__",
        "//tensorflow:internal",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "structured",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    tags = ["nofixdeps"],
    deps = [
        ":structured_tensor",
    ],
)

py_library(
    name = "structured_tensor",
    srcs = ["structured_tensor.py"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

py_test(
    name = "structured_tensor_test",
    srcs = ["structured_tensor_test.py"],
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
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
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
