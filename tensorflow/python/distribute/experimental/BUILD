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
        "//tensorflow/python/distribute/failure_handling:preemption_watcher",
    ],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":dtensor_util",
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:input_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util",
    ],
)

dtensor_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    shard_count = {"tpu": 2},
    tags = ["no_pip"],
    deps = [
        ":dtensor_util",
        ":mirrored_strategy",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/dtensor/python/tests:test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "dtensor_util",
    srcs = ["dtensor_util.py"],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/python/distribute:values",
    ],
)

dtensor_test(
    name = "dtensor_util_test",
    srcs = ["dtensor_util_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dtensor_util",
        ":mirrored_strategy",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python/tests:test_util",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
