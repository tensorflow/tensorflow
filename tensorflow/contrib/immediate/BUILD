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
        "python/immediate/ops_helper.py",
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
    data = [
         "python/immediate/ops_history.v0.pbtxt"
    ]
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
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:extra_py_tests_deps",
    ],
)

py_test(
    name = "env_test2",
    size = "small",
    srcs = ["python/immediate/env_test2.py"],
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
    name = "ops_helper_test",
    size = "small",
    srcs = ["python/immediate/ops_helper_test.py"],
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


py_test(
    name = "math_ops_test",
    srcs = ["python/immediate/ops/math_ops_test.py"],
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

# py_test(
#     name = "histogram_ops_test",
#     srcs = ["python/immediate/ops/histogram_ops_test.py"],
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
#     name = "math_grad_test",
#     srcs = ["python/immediate/ops/math_grad_test.py"],
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
#     name = "special_math_ops_test",
#     srcs = ["python/immediate/ops/special_math_ops_test.py"],
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

py_test(
    name = "nn_test",
    srcs = ["python/immediate/ops/nn_test.py"],
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

#cuda_py_tests(
#    name = "op_tests",
#    size = "small",
##    srcs = ["python/immediate/ops/math_ops_test.py"]
#    srcs = glob(
#        ["python/immediate/ops/*_test.py"],
#        exclude = [
#            "python/immediate/ops/image_ops_test.py",
#            "python/immediate/ops/nn_test.py",
#            "python/immediate/ops/op_def_library_test.py",
#            "python/immediate/ops/math_ops_test.py",
#            "python/immediate/ops/histogram_ops_test.py",
#            "python/immediate/ops/math_grad_test.py",
#            "python/immediate/ops/special_math_ops_test",
#        ],
#    ),
#)

#cuda_py_tests(
#    name = "nn_test",
#    srcs = ["python/immediate/nn_test.py"]
#)

#cuda_py_tests(
#    name = "math_ops_test",
#    srcs = ["python/immediate/ops/math_ops_test.py"]
#)

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
