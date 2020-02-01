# Description:
#   Contains the Keras engine API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "engine",
    srcs = [
        "__init__.py",
        "compile_utils.py",
        "input_layer.py",
        "network.py",
        "node.py",
        "partial_batch_padding_handler.py",
        "saving.py",
        "sequential.py",
        "training.py",
        "training_arrays.py",
        "training_distributed.py",
        "training_eager.py",
        "training_generator.py",
        "training_utils.py",
        "training_v1.py",
        "training_v2.py",
        "training_v2_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer",
        ":base_preprocessing_layer",
        ":data_adapter",
        ":input_spec",
        "//tensorflow/python:composite_tensor_utils",
        "//tensorflow/python:py_checkpoint_reader",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:callbacks_v1",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/keras/mixed_precision/experimental:autocast_variable",
        "//tensorflow/python/keras/mixed_precision/experimental:loss_scale_optimizer",
        "//tensorflow/python/keras/mixed_precision/experimental:policy",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/keras/utils:version_utils",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_util",
        "//tensorflow/python/profiler:traceme",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_library(
    name = "base_layer_utils",
    srcs = ["base_layer_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:auto_control_deps",
        "//tensorflow/python:control_flow_v2_func_graphs",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_layer",
    srcs = ["base_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_layer_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/keras/mixed_precision/experimental:autocast_variable",
        "//tensorflow/python/keras/mixed_precision/experimental:loss_scale_optimizer",
        "//tensorflow/python/keras/mixed_precision/experimental:policy",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/module",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_library(
    name = "data_adapter",
    srcs = ["data_adapter.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "input_spec",
    srcs = ["input_spec.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras:backend",
        "@six_archive//:six",
    ],
)

py_library(
    name = "base_preprocessing_layer",
    srcs = [
        "base_preprocessing_layer.py",
        "base_preprocessing_layer_v1.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:composite_tensor_utils",
        "//tensorflow/python/data",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/module",
    ],
)

tf_py_test(
    name = "base_layer_utils_test",
    srcs = ["base_layer_utils_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer_utils",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "data_adapter_test",
    size = "medium",
    srcs = ["data_adapter_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":data_adapter",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "base_preprocessing_layer_test",
    size = "medium",
    srcs = ["base_preprocessing_layer_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_preprocessing_layer",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "training_gpu_test",
    size = "small",
    srcs = ["training_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "correctness_test",
    size = "medium",
    srcs = ["correctness_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "input_spec_test",
    size = "small",
    srcs = ["input_spec_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_test",
    size = "medium",
    srcs = ["training_test.py"],
    python_version = "PY3",
    shard_count = 20,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "compile_utils_test",
    size = "medium",
    srcs = ["compile_utils_test.py"],
    tags = [
        "nomac",  # TODO(b/146226927)
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_dataset_test",
    size = "medium",
    srcs = ["training_dataset_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_arrays_test",
    size = "medium",
    srcs = ["training_arrays_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/layers",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "training_generator_test",
    size = "medium",
    srcs = ["training_generator_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # TODO(b/132183295): Re-enable this.
        "nomac",  # TODO(b/140193633): Re-enable this.
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_integration_test",
    size = "medium",
    srcs = ["training_integration_test.py"],
    python_version = "PY3",
    shard_count = 30,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "feature_columns_integration_test",
    size = "medium",
    srcs = ["feature_columns_integration_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_eager_test",
    size = "medium",
    srcs = ["training_eager_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_utils_test",
    size = "medium",
    srcs = ["training_utils_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_v2_utils_test",
    size = "medium",
    srcs = ["training_v2_utils_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "network_test",
    size = "medium",
    srcs = ["network_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no-internal-py3",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "base_layer_test",
    size = "medium",
    srcs = ["base_layer_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "control_flow_test",
    size = "medium",
    srcs = ["control_flow_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "medium",
    srcs = ["sequential_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
