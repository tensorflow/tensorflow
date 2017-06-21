# Description: Operations defined for Cloud TPUs

package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

cc_library(
    name = "all_ops",
    deps = [
        ":cross_replica_ops_op_lib",
        ":infeed_ops_op_lib",
        ":outfeed_ops_op_lib",
        ":replication_ops_op_lib",
        ":tpu_configuration_ops_op_lib",
        ":tpu_sendrecv_ops_op_lib",
    ],
)

py_library(
    name = "tpu_estimator",
    srcs = [
        "python/tpu/tpu_config.py",
        "python/tpu/tpu_estimator.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu",
        ":tpu_py",
        ":training_loop",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "cross_replica_ops",
        "infeed_ops",
        "outfeed_ops",
        "replication_ops",
        "tpu_configuration_ops",
        "tpu_sendrecv_ops",
    ],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_custom_op_library(
    name = "python/ops/_tpu_ops.so",
    srcs = [
        "ops/cross_replica_ops.cc",
        "ops/infeed_ops.cc",
        "ops/outfeed_ops.cc",
        "ops/replication_ops.cc",
        "ops/tpu_configuration_ops.cc",
        "ops/tpu_sendrecv_ops.cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "tpu_ops",
    deps = [
        ":cross_replica_ops_op_lib",
        ":infeed_ops_op_lib",
        ":outfeed_ops_op_lib",
        ":replication_ops_op_lib",
        ":tpu_configuration_ops_op_lib",
        ":tpu_sendrecv_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "tpu_py",
    srcs = glob(["python/ops/*.py"]) + ["__init__.py"],
    dso = [":python/ops/_tpu_ops.so"],
    kernels = [
        ":all_ops",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "tpu_helper_library",
    srcs_version = "PY2AND3",
    deps = [
        ":tpu",
        ":tpu_feed",
        ":tpu_function",
        ":tpu_py",
        ":tpu_sharding",
        ":training_loop",
    ],
)

py_library(
    name = "tpu_function",
    srcs = ["python/tpu/tpu_function.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_feed",
        ":tpu_py",
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "tpu",
    srcs = [
        "python/tpu/__init__.py",
        "python/tpu/tpu.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_py",
        ":training_loop",
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "tpu_sharding",
    srcs = ["python/tpu/tpu_sharding.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "tpu_feed",
    srcs = ["python/tpu/tpu_feed.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_py",
        ":tpu_sharding",
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "training_loop",
    srcs = [
        "python/tpu/tpu_optimizer.py",
        "python/tpu/training_loop.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_function",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "tpu_sharding_test",
    size = "small",
    srcs = ["python/tpu/tpu_sharding_test.py"],
    additional_deps = [
        ":tpu_sharding",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
    ],
)

tf_py_test(
    name = "tpu_infeed_test",
    size = "small",
    srcs = ["python/tpu/tpu_infeed_test.py"],
    additional_deps = [
        ":tpu_feed",
        ":tpu_sharding",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
    ],
)

tf_py_test(
    name = "tpu_function_test",
    size = "small",
    srcs = ["python/tpu/tpu_function_test.py"],
    additional_deps = [
        ":tpu_function",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
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
