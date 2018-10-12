# Description:
#   Contains ops for working with statistical distributions,
#   particularly useful for Bayesian inference.
#   APIs here are meant to evolve over time.

package(default_visibility = [
    "//learning/brain/contrib/bayesflow:__subpackages__",
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "bayesflow_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "monte_carlo_test",
    size = "small",
    srcs = ["python/kernel_tests/monte_carlo_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
    ],
)
