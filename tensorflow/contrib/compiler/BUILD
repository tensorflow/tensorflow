licenses(["notice"])  # Apache 2.0

package(default_visibility = [":friends"])

package_group(
    name = "friends",
    includes = ["//tensorflow/compiler/jit:friends"],
    packages = ["//tensorflow/..."],
)

load("//tensorflow:tensorflow.bzl", "gpu_py_test")

py_library(
    name = "compiler_py",
    srcs = [
        "__init__.py",
        "jit.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

gpu_py_test(
    name = "jit_test",
    size = "small",
    srcs = ["jit_test.py"],
    additional_deps = [
        ":compiler_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    xla_enabled = True,
)
