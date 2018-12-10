package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nccl_ops",
    ],
)

tf_py_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.py"],
    additional_deps = [
        ":all_reduce",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
    ],
)

py_library(
    name = "cross_device_ops",
    srcs = ["cross_device_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cross_device_utils",
        ":device_util",
        ":reduce_util",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:device_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

py_library(
    name = "cross_device_utils",
    srcs = ["cross_device_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":all_reduce",
        ":values",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nccl_ops",
    ],
)

py_library(
    name = "device_util",
    srcs = ["device_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:device",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "device_util_test",
    srcs = ["device_util_test.py"],
    additional_deps = [
        ":device_util",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
    ],
)

py_library(
    name = "distribute",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_lib",
        ":mirrored_strategy",
    ],
)

py_library(
    name = "distribute_lib",
    srcs = [
        "distribute_lib.py",
        "distribution_strategy_context.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":device_util",
        ":reduce_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/ops/losses",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_test(
    name = "distribute_lib_test",
    size = "small",
    srcs = ["distribute_lib_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_lib",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "distribute_config",
    srcs = [
        "distribute_config.py",
    ],
    deps = [],
)

py_library(
    name = "distribute_coordinator",
    srcs = [
        "distribute_coordinator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_coordinator_context",
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "distribute_coordinator_test",
    srcs = ["distribute_coordinator_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_pip",
    ],
    deps = [
        ":distribute_coordinator",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distributed_framework_test_lib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "distribute_coordinator_context",
    srcs = [
        "distribute_coordinator_context.py",
    ],
    srcs_version = "PY2AND3",
    deps = [],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":cross_device_ops",
        ":device_util",
        ":distribute_lib",
        ":multi_worker_util",
        ":reduce_util",
        ":shared_variable_creator",
        ":values",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:tape",
    ],
)

py_library(
    name = "multi_worker_util",
    srcs = [
        "multi_worker_util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "input_ops",
    srcs = ["input_ops.py"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/ops:filter_for_shard_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

cuda_py_test(
    name = "input_ops_test",
    srcs = ["input_ops_test.py"],
    additional_deps = [
        ":input_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python:errors",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
    ],
    tags = [
        "no_pip",
    ],
)

py_test(
    name = "multi_worker_util_test",
    srcs = ["multi_worker_util_test.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        ":multi_worker_util",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

# Used only by estimator.
py_library(
    name = "estimator_training",
    srcs = [
        "estimator_training.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":distribute_coordinator",
        ":distribute_coordinator_context",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "reduce_util",
    srcs = ["reduce_util.py"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "shared_variable_creator",
    srcs = ["shared_variable_creator.py"],
)

py_test(
    name = "shared_variable_creator_test",
    srcs = ["shared_variable_creator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":shared_variable_creator",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    deps = [
        ":device_util",
        ":distribute_lib",
        ":input_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/checkpointable:base",
        "@six_archive//:six",
    ],
)
