licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/jit:friends"],
    packages = [
        "//tensorflow/...",
        "//tensorflow_models/...",
        "//third_party/py/tensor2tensor/...",
    ],
)

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":xla",
        "//tensorflow/python/compiler/xla:compiler_py",
    ],
)

py_library(
    name = "xla",
    srcs = ["xla.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/compiler/xla:compiler_py",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

cuda_py_test(
    name = "xla_test",
    srcs = ["xla_test.py"],
    additional_deps = [
        ":xla",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/tpu:tpu_estimator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_mac",
        "no_windows",
    ],
    xla_enabled = True,
)
