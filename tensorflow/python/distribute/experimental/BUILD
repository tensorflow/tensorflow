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

py_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dtensor_test_util",
        ":mirrored_strategy",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dtensor_test_util",
    srcs = ["dtensor_test_util.py"],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:device_spec",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
