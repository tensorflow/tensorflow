load(
    "//tensorflow/dtensor:build_defs.bzl",
    "dtensor_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/distribute:central_storage_strategy",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/failure_handling:failure_handling_lib",
    ],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

dtensor_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    tags = ["no_pip"],
    deps = [
        ":mirrored_strategy",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python/tests:test_util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
