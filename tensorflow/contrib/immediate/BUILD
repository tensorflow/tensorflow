# Description:
#   Contains implementation of immediate execution in TensorFlow
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "immediate_py",
    srcs = [
         "__init__.py",
         "python/immediate/env.py",
         "python/immediate/itensor.py",
         "python/immediate/module_rewriter.py",
         "python/immediate/op.py",
         "python/immediate/test_util.py",
         "python/immediate/util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
         "//tensorflow/python:client",
         "//tensorflow/python:framework",
         "//tensorflow/python:lib",
         "//tensorflow/python:ops",
         "//tensorflow/python:pywrap_tensorflow",
    ],
)

py_library(
    name = "immediate_test_lib",
    srcs = [],
    srcs_version = "PY2AND3",
    deps = [
	"//tensorflow/python:framework_test_lib",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow:tensorflow_py",
        ":immediate_py",
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
    name = "env_cache_test",
    size = "small",
    srcs = ["python/immediate/env_cache_test.py"],
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
    name = "itensor_test",
    size = "small",
    srcs = ["python/immediate/itensor_test.py"],
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
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:framework_test_lib",
#           "//tensorflow/python:kernel_tests/gradient_checker",
           "//tensorflow/python:platform_test",
           "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "batchnorm_test",
    srcs = ["python/immediate/extra_tests/batchnorm_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow/python:extra_py_tests_deps",
           "//tensorflow/python:framework_test_lib",
#           "//tensorflow/python:kernel_tests/gradient_checker",
           "//tensorflow/python:platform_test",
           "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "image_ops_test",
    srcs = ["python/immediate/extra_tests/image_ops_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
    ],

    data = [
        "//tensorflow/core:image_testdata",
    ],
)

py_test(
    name = "image_ops_double_test",
    srcs = ["python/immediate/extra_tests/image_ops_double_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
    ],

    data = [
        "//tensorflow/core:image_testdata",
    ],
)

py_test(
    name = "histogram_ops_test",
    srcs = ["python/immediate/extra_tests/histogram_ops_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
#           "//tensorflow/python:kernel_tests/gradient_checker",
    ],
)


py_test(
    name = "nn_test",
    srcs = ["python/immediate/extra_tests/nn_test.py"],
    size = "small",
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/python:framework_test_lib",
           "//tensorflow/python:platform_test",
           "//tensorflow/python:extra_py_tests_deps",
#           "//tensorflow/python:kernel_tests/gradient_checker",
    ],
)

py_test(
    name = "mnist_inference_test",
    srcs = ["python/immediate/mnist_inference_test.py"],
    size = "medium",
    srcs_version = "PY2AND3",
    deps = [
           ":immediate_py",
           ":immediate_test_lib",
           "//tensorflow:tensorflow_py",
           "//tensorflow/models/image/mnist:convolutional",
    ],

    data = [
    	 ":mnist_testdata"
    ],
)

filegroup(
    name = "mnist_testdata",
    srcs = [
    	"python/immediate/testdata/checkpoint",
	"python/immediate/testdata/convolutional-0",
	"python/immediate/testdata/convolutional-0.meta",
	"python/immediate/testdata/t10k-images-idx3-ubyte.gz",
        "python/immediate/testdata/t10k-labels-idx1-ubyte.gz",
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
