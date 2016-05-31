# Description:
#   Contains implementation of immediate execution in TensorFlow
#   APIs here are meant to evolve over time.

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "immediate_py",
    srcs = [
        "python/immediate/env.py",
        "python/immediate/tensor.py",
        "python/immediate/op.py",
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
        "//tensorflow:tensorflow_py",
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
    ],
)

py_test(
    name = "env_extra_test",
    size = "small",
    srcs = ["python/immediate/env_extra_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
	":immediate_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "module_rewriter_test",
    size = "small",
    srcs = ["python/immediate/module_rewriter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        ":immediate_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensor_test",
    size = "small",
    srcs = ["python/immediate/tensor_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
        ":immediate_test_lib",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tensor_handle_test",
    size = "small",
    srcs = ["python/immediate/tensor_handle_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":immediate_py",
	":immediate_test_lib",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "math_ops_test",
    srcs = ["python/immediate/extra_tests/math_ops_test.py"],
    size = "small",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:kernel_tests/gradient_checker",
           ],
)

py_test(
    name = "nn_test",
    srcs = ["python/immediate/extra_tests/nn_test.py"],
    size = "medium",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:kernel_tests/gradient_checker",
           ],
)


py_test(
    name = "batchnorm_test",
    srcs = ["python/immediate/extra_tests/batchnorm_test.py"],
    size = "medium",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:kernel_tests/gradient_checker",
           ],
)

py_test(
    name = "image_ops_test",
    srcs = ["python/immediate/extra_tests/image_ops_test.py"],
    size = "medium",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           ],

    data = [
        "//tensorflow/core:image_testdata",
    ],
)

# TODO(yaroslavvb): tests below use ops that utilize ContextManager
# constructs like op_scope and ops.colocate_with. Re-enable them once
# ContextManager closures are copied in module_rewriter.
# 
# py_test(
#     name = "special_math_ops_test",
#     srcs = ["python/immediate/extra_tests/special_math_ops_test.py"],
#     deps = [
#            ":immediate_py",
#            ":immediate_test_lib",
#            "//tensorflow:tensorflow_py",
#            "//tensorflow/python:framework_test_lib",
#            "//tensorflow/python:platform_test",
#            "//tensorflow/python:extra_py_tests_deps",
#            "//tensorflow/python:kernel_tests/gradient_checker",
#            ],
# )

# py_test(
#     name = "nn_test2",
#     srcs = ["python/immediate/extra_tests/nn_test2.py"],
#     size = "medium",
#     deps = [
#            ":immediate_py",
#            ":immediate_test_lib",
#            "//tensorflow:tensorflow_py",
#            "//tensorflow/python:framework_test_lib",
#            "//tensorflow/python:platform_test",
#            "//tensorflow/python:extra_py_tests_deps",
#            "//tensorflow/python:kernel_tests/gradient_checker",
#            ],
# )

# py_test(
#     name = "control_flow_ops_test",
#     srcs = ["python/immediate/extra_tests/control_flow_ops_test.py"],
#     size = "medium",
#     deps = [
#            ":immediate_py",
#            ":immediate_test_lib",
#            "//tensorflow:tensorflow_py",
#            "//tensorflow/python:framework_test_lib",
#            "//tensorflow/python:platform_test",
#            "//tensorflow/python:extra_py_tests_deps",
#            "//tensorflow/python:kernel_tests/gradient_checker",
#            ],
# )
#
# py_test(
#     name = "histogram_ops_test",
#     srcs = ["python/immediate/extra_tests/histogram_ops_test.py"],
#     size = "medium",
#     deps = [
#            ":immediate_py",
#            ":immediate_test_lib",
#            "//tensorflow:tensorflow_py",
#            "//tensorflow/python:framework_test_lib",
#            "//tensorflow/python:platform_test",
#            "//tensorflow/python:extra_py_tests_deps",
#            "//tensorflow/python:kernel_tests/gradient_checker",
#            ],
# )




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
