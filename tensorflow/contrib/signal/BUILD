package(default_visibility = ["//tensorflow:__subpackages__"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow:tensorflow.bzl", "py_test")  # @unused

py_library(
    name = "signal_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:spectral_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "test_util",
    srcs = ["python/kernel_tests/test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python:training",
    ],
)

cuda_py_tests(
    name = "mel_ops_test",
    srcs = ["python/kernel_tests/mel_ops_test.py"],
    additional_deps = [
        ":signal_py",
        ":test_util",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_tests(
    name = "mfcc_ops_test",
    srcs = ["python/kernel_tests/mfcc_ops_test.py"],
    additional_deps = [
        ":signal_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:spectral_ops_test_util",
    ],
)

cuda_py_tests(
    name = "reconstruction_ops_test",
    srcs = ["python/kernel_tests/reconstruction_ops_test.py"],
    additional_deps = [
        ":signal_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "shape_ops_test",
    srcs = ["python/kernel_tests/shape_ops_test.py"],
    additional_deps = [
        ":signal_py",
        ":test_util",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "spectral_ops_test",
    size = "large",
    srcs = ["python/kernel_tests/spectral_ops_test.py"],
    additional_deps = [
        ":signal_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:spectral_ops_test_util",
    ],
    tags = ["nomac"],
)

cuda_py_tests(
    name = "window_ops_test",
    srcs = ["python/kernel_tests/window_ops_test.py"],
    additional_deps = [
        ":signal_py",
        ":test_util",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
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
    visibility = ["//tensorflow:__subpackages__"],
)
