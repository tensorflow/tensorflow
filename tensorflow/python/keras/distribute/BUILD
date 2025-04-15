# Description:
#   keras/distribute package is intended to serve as the centralized place for things
#   related to dist-strat used by Keras..

load("//tensorflow:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # TODO(scottzhu): Remove this deps when distribute test are converted to integration test.
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "distribute",
    srcs = [
        "__init__.py",
        "distributed_training_utils.py",
        "distributed_training_utils_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distribute_coordinator_utils",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:callbacks_v1",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/mixed_precision:policy",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "optimizer_combinations",
    srcs = ["optimizer_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/training",
    ],
)

py_library(
    name = "worker_training_state",
    srcs = [
        "worker_training_state.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":distributed_file_utils",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/training:checkpoint_management",
    ],
)

py_library(
    name = "distributed_file_utils",
    srcs = [
        "distributed_file_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/lib/io:file_io",
    ],
)

py_library(
    name = "distribute_coordinator_utils",
    srcs = [
        "distribute_coordinator_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:server_lib",
    ],
)
