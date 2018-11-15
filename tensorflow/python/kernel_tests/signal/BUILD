package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_tests")
load("//tensorflow:tensorflow.bzl", "py_test")  # @unused

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf_optimizer",
        "//tensorflow/python:training",
    ],
)

cuda_py_tests(
    name = "dct_ops_test",
    srcs = ["dct_ops_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:spectral_ops_test_util",
        "//tensorflow/python/ops/signal",
    ],
)

cuda_py_tests(
    name = "fft_ops_test",
    size = "medium",
    srcs = ["fft_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:spectral_ops_test_util",
        "//tensorflow/python/ops/signal",
    ],
    shard_count = 4,
    tags = ["optonly"],
)

cuda_py_tests(
    name = "mel_ops_test",
    srcs = ["mel_ops_test.py"],
    additional_deps = [
        ":test_util",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/ops/signal",
    ],
)

cuda_py_tests(
    name = "mfcc_ops_test",
    srcs = ["mfcc_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python:spectral_ops_test_util",
    ],
)

cuda_py_tests(
    name = "reconstruction_ops_test",
    srcs = ["reconstruction_ops_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "shape_ops_test",
    srcs = ["shape_ops_test.py"],
    additional_deps = [
        ":test_util",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_tests(
    name = "spectral_ops_test",
    size = "large",
    srcs = ["spectral_ops_test.py"],
    additional_deps = [
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
        "//tensorflow/python/ops/signal",
    ],
    tags = ["nomac"],
)

cuda_py_tests(
    name = "window_ops_test",
    srcs = ["window_ops_test.py"],
    additional_deps = [
        ":test_util",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/ops/signal",
        "//tensorflow/python:platform_test",
    ],
)
