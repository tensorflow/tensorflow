# Description:
#   keras/distribute package is intended to serve as the centralized place for things
#   related to dist-strat used by Keras..

package(
    # TODO(scottzhu): Remove this deps when distribute test are converted to integration test.
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
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
        ":sidecar_evaluator",
        "//tensorflow/python:client_testlib",
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
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "distribute_test_lib_pip",
    srcs_version = "PY3",
    deps = [
        ":multi_worker_testing_utils",
    ],
)

py_library(
    name = "optimizer_combinations",
    srcs = ["optimizer_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:training",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/keras/optimizer_v2",
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
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/training:checkpoint_management",
        "//tensorflow/python/training/tracking:util",
    ],
)

py_library(
    name = "multi_worker_testing_utils",
    srcs = [
        "multi_worker_testing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/optimizer_v2",
    ],
)

py_library(
    name = "tpu_strategy_test_utils",
    srcs = ["tpu_strategy_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/tpu:tpu_strategy_util",
    ],
)

py_library(
    name = "distributed_file_utils",
    srcs = [
        "distributed_file_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:lib",
        "//tensorflow/python/distribute:distribute_lib",
    ],
)

py_library(
    name = "sidecar_evaluator",
    srcs = ["sidecar_evaluator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/training/tracking:util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "strategy_combinations",
    srcs = ["strategy_combinations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/distribute:strategy_combinations",
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
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/distribute:multi_worker_util",
    ],
)
