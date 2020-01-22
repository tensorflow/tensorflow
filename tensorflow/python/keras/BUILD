# Description:
#   Contains the Keras API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "estimator/__init__.py",
        "keras_parameterized.py",
        "ops.py",
        "testing_utils.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":backend",
        ":engine",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras/applications",
        "//tensorflow/python/keras/datasets",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/mixed_precision/experimental:mixed_precision_experimental",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/premade",
        "//tensorflow/python/keras/preprocessing",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils",
        "//tensorflow/python/keras/wrappers",
        "//tensorflow/python/saved_model",
    ],
)

py_library(
    name = "backend",
    srcs = ["backend.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:composite_tensor_utils",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:ctc_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:multi_worker_util",
    ],
)

py_library(
    name = "backend_config",
    srcs = ["backend_config.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "base_layer_utils",
    srcs = ["engine/base_layer_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
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
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "engine",
    srcs = [
        "engine/__init__.py",
        "engine/compile_utils.py",
        "engine/input_layer.py",
        "engine/network.py",
        "engine/node.py",
        "engine/partial_batch_padding_handler.py",
        "engine/saving.py",
        "engine/sequential.py",
        "engine/training.py",
        "engine/training_arrays.py",
        "engine/training_distributed.py",
        "engine/training_eager.py",
        "engine/training_generator.py",
        "engine/training_utils.py",
        "engine/training_v1.py",
        "engine/training_v2.py",
        "engine/training_v2_utils.py",
        "metrics.py",  # Need base_layer
        "models.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
        ":backend",
        ":base_layer",
        ":base_preprocessing_layer",
        ":callbacks",
        ":callbacks_v1",
        ":constraints",
        ":data_adapter",
        ":initializers",
        ":input_spec",
        ":losses",
        ":optimizers",
        ":regularizers",
        "//tensorflow/python:composite_tensor_utils",
        "//tensorflow/python:py_checkpoint_reader",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:monitoring",
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
    name = "data_adapter",
    srcs = ["engine/data_adapter.py"],
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
    srcs = ["engine/input_spec.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "@six_archive//:six",
    ],
)

py_library(
    name = "base_layer",
    srcs = ["engine/base_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":base_layer_utils",
        ":constraints",
        ":regularizers",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:monitoring",
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
    name = "base_preprocessing_layer",
    srcs = [
        "engine/base_preprocessing_layer.py",
        "engine/base_preprocessing_layer_v1.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":base_layer",
        "//tensorflow/python:composite_tensor_utils",
        "//tensorflow/python/data",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/module",
    ],
)

py_library(
    name = "activations",
    srcs = [
        "activations.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "callbacks",
    srcs = [
        "callbacks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/distribute:distributed_file_utils",
        "//tensorflow/python/keras/distribute:multi_worker_training_state",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "callbacks_v1",
    srcs = [
        "callbacks_v1.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/eager:profiler",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "constraints",
    srcs = [
        "constraints.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "initializers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "losses",
    srcs = [
        "losses.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "optimizers",
    srcs = [
        "optimizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "regularizers",
    srcs = [
        "regularizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

tf_py_test(
    name = "integration_test",
    size = "medium",
    srcs = ["integration_test.py"],
    python_version = "PY3",
    shard_count = 16,
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "constraints_test",
    size = "small",
    srcs = ["constraints_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "data_adapter_test",
    size = "medium",
    srcs = ["engine/data_adapter_test.py"],
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
    name = "initializers_test",
    size = "small",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:init_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "regularizers_test",
    size = "medium",
    srcs = ["regularizers_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["optimizers_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "losses_test",
    size = "small",
    srcs = ["losses_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "add_loss_correctness_test",
    size = "medium",
    srcs = ["add_loss_correctness_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "metrics_functional_test",
    size = "small",
    srcs = ["metrics_functional_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "metrics_confusion_matrix_test",
    size = "medium",
    srcs = ["metrics_confusion_matrix_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "metrics_correctness_test",
    size = "medium",
    srcs = ["metrics_correctness_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "temporal_sample_weights_correctness_test",
    size = "medium",
    srcs = ["temporal_sample_weights_correctness_test.py"],
    python_version = "PY3",
    shard_count = 20,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "base_preprocessing_layer_test",
    size = "medium",
    srcs = ["engine/base_preprocessing_layer_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "training_gpu_test",
    size = "small",
    srcs = ["engine/training_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "callbacks_test",
    size = "medium",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "callbacks_v1_test",
    size = "medium",
    srcs = ["callbacks_v1_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "correctness_test",
    size = "medium",
    srcs = ["engine/correctness_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "input_spec_test",
    size = "small",
    srcs = ["engine/input_spec_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_test",
    size = "medium",
    srcs = ["engine/training_test.py"],
    python_version = "PY3",
    shard_count = 20,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "compile_utils_test",
    size = "medium",
    srcs = ["engine/compile_utils_test.py"],
    tags = [
        "nomac",  # TODO(b/146226927)
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_dataset_test",
    size = "medium",
    srcs = ["engine/training_dataset_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_arrays_test",
    size = "medium",
    srcs = ["engine/training_arrays_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/layers",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "training_generator_test",
    size = "medium",
    srcs = ["engine/training_generator_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # TODO(b/132183295): Re-enable this.
        "nomac",  # TODO(b/140193633): Re-enable this.
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_integration_test",
    size = "medium",
    srcs = ["engine/training_integration_test.py"],
    python_version = "PY3",
    shard_count = 30,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "feature_columns_integration_test",
    size = "medium",
    srcs = ["engine/feature_columns_integration_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/feature_column:feature_column_py",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_eager_test",
    size = "medium",
    srcs = ["engine/training_eager_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_utils_test",
    size = "medium",
    srcs = ["engine/training_utils_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "training_v2_utils_test",
    size = "medium",
    srcs = ["engine/training_v2_utils_test.py"],
    python_version = "PY3",
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:strategy_combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "model_subclassing_test_util",
    srcs = ["model_subclassing_test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
    ],
)

tf_py_test(
    name = "model_subclassing_test",
    size = "medium",
    srcs = ["model_subclassing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",
        "notsan",
    ],
    deps = [
        ":keras",
        ":model_subclassing_test_util",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "model_subclassing_compiled_test",
    size = "medium",
    srcs = ["model_subclassing_compiled_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",
        "notsan",
    ],
    deps = [
        ":keras",
        ":model_subclassing_test_util",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "custom_training_loop_test",
    size = "medium",
    srcs = ["custom_training_loop_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "network_test",
    size = "medium",
    srcs = ["engine/network_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no-internal-py3",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "base_layer_test",
    size = "medium",
    srcs = ["engine/base_layer_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "base_layer_utils_test",
    srcs = ["engine/base_layer_utils_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "control_flow_test",
    size = "medium",
    srcs = ["engine/control_flow_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "medium",
    srcs = ["engine/sequential_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "models_test",
    size = "medium",
    srcs = ["models_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # b/67509773
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "backend_test",
    size = "medium",
    srcs = ["backend_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "backend_config_test",
    size = "medium",
    srcs = ["backend_config_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "keras_parameterized_test",
    size = "small",
    srcs = ["keras_parameterized_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
