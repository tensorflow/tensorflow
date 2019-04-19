# Description: Operations defined for Cloud TPUs

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
    "tf_py_test",
)
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//cloud/vmm/testing/tests/tpu:__subpackages__",
        "//knowledge/cerebra/sense/im2query:__subpackages__",
        "//learning/brain:__subpackages__",
        "//learning/deepmind:__subpackages__",
        "//medical/pathology:__subpackages__",
        "//smartass/brain:__subpackages__",
        "//tensorflow:__subpackages__",
        "//tensorflow_models:__subpackages__",
        "//vr/perception:__subpackages__",
    ],
)

py_library(
    name = "tpu_py",
    srcs = ["python/ops/tpu_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/tpu:tpu_py",
    ],
)

py_library(
    name = "async_checkpoint",
    srcs = ["python/tpu/async_checkpoint.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/tpu:async_checkpoint",
    ],
)

py_library(
    name = "tpu_estimator",
    srcs = [
        "python/tpu/_tpu_estimator_embedding.py",
        "python/tpu/error_handling.py",
        "python/tpu/tpu_config.py",
        "python/tpu/tpu_context.py",
        "python/tpu/tpu_estimator.py",
        "python/tpu/util.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":async_checkpoint",
        ":feature_column",
        ":functional",
        ":tpu_embedding",
        ":tpu_lib",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/python/tpu:tpu_estimator",
    ],
)

py_library(
    name = "functional",
    srcs = ["python/tpu/functional.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/python/tpu:functional",
    ],
)

py_library(
    name = "profiler",
    srcs = ["python/profiler/__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/tpu/profiler",
    ],
)

py_library(
    name = "tpu",
    srcs = [
        "__init__.py",
        "python/tpu/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_column",
        ":keras_support",  # split out to avoid cycle with tpu_strategy
        ":tpu_embedding",
        ":tpu_estimator",
        ":tpu_lib",
        "//tensorflow/python/tpu",
    ],
)

py_library(
    name = "keras_support",
    srcs = [
        "python/tpu/keras_support.py",
        "python/tpu/keras_tpu_variables.py",
    ],
    srcs_version = "PY2AND3",
    visibility = [
        "//cloud/vmm/testing/tests/tpu:__subpackages__",
        "//learning/brain:__subpackages__",
        "//tensorflow:__subpackages__",
        "//third_party/cloud_tpu/models/keras_colab:__subpackages__",
        "//third_party/cloud_tpu/models/resnet50_keras:__subpackages__",
    ],
    deps = [
        ":tpu_lib",
        "//tensorflow/contrib/cluster_resolver:cluster_resolver_py",
        "//tensorflow/contrib/distribute",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/protobuf/tpu:compilation_result_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "tpu_lib",
    srcs = [
        "python/tpu/__init__.py",
        "python/tpu/bfloat16.py",
        "python/tpu/device_assignment.py",
        "python/tpu/session_support.py",
        "python/tpu/tensor_tracer.py",
        "python/tpu/topology.py",
        "python/tpu/tpu.py",
        "python/tpu/tpu_feed.py",
        "python/tpu/tpu_function.py",
        "python/tpu/tpu_optimizer.py",
        "python/tpu/tpu_sharding.py",
        "python/tpu/tpu_system_metadata.py",
        "python/tpu/training_loop.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":datasets",
        ":functional",
        ":profiler",
        ":tpu_py",
        "//tensorflow/contrib/cluster_resolver:cluster_resolver_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python/tpu:tpu_lib",
    ],
)

py_library(
    name = "datasets",
    srcs = [
        "python/tpu/datasets.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/tpu:datasets",
    ],
)

py_library(
    name = "tpu_embedding",
    srcs = [
        "python/tpu/tpu_embedding.py",
        "python/tpu/tpu_embedding_gradient.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_lib",
        "//tensorflow/python/tpu:tpu_embedding",
    ],
)

py_library(
    name = "feature_column",
    srcs = ["python/tpu/feature_column.py"],
    deps = [
        ":tpu_lib",
        "//tensorflow/python/tpu:feature_column",
    ],
)
