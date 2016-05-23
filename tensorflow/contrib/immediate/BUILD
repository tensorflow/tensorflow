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

py_library(
    name = "immediate_test_lib",
    srcs = [
        "python/immediate/test_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
	"//tensorflow/python:framework_test_lib",
    ],
)


py_test(
    name = "env_test",
    size = "small",
    srcs = ["python/immediate/env_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
	":immediate_test_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:extra_py_tests_deps",
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
