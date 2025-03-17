load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "mnist_testing_utils",
    srcs = ["mnist_testing_utils.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:extra_py_tests_deps",
    ],
)

cuda_py_strict_test(
    name = "profiler_api_test",
    srcs = ["profiler_api_test.py"],
    tags = [
        "no_oss",  # TODO(b/283175845)
        "no_pip",
        "no_windows",  # TODO(b/192257727)
    ],
    deps = [
        ":mnist_testing_utils",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/profiler:profiler_client",
        "//tensorflow/python/profiler:profiler_v2",
        "@pypi_portpicker//:pkg",
    ],
)
