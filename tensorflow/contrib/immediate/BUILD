# Description:
#   Contains implementation of immediate execution in TensorFlow
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "immediate_py",
    srcs = [
        "python/immediate/env.py",
        "python/immediate/tensor.py",
        "python/immediate/op.py",
        "python/immediate/wrapping_util.py",
        "python/immediate/wrapping_manager.py",
	"python/immediate/module_rewriter.py",
        "__init__.py",
        "python/__init__.py",
        "python/immediate/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
         "//tensorflow/python:framework",
         "//tensorflow/python:client",
         "//tensorflow/python:lib",
         "//tensorflow/python:ops",
         "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_test(
    name = "env_test",
    size = "small",
    srcs = ["python/immediate/env_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "module_rewriter_test",
    size = "small",
    srcs = ["python/immediate/module_rewriter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)


py_test(
    name = "wrapping_util_test",
    size = "small",
    srcs = ["python/immediate/wrapping_util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "wrapping_manager_test",
    size = "small",
    srcs = ["python/immediate/wrapping_manager_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "env_test2",
    size = "small",
    srcs = ["python/immediate/env_test2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "tensor_test",
    size = "small",
    srcs = ["python/immediate/tensor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "op_test",
    size = "small",
    srcs = ["python/immediate/op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
