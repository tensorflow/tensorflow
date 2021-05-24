# Description:
#   Contains the Keras engine API (internal TensorFlow version).

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/feature_column:__pkg__",
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
    name = "engine",
    srcs = [
        "__init__.py",
        "compile_utils.py",
        "functional.py",
        "input_layer.py",
        "partial_batch_padding_handler.py",
        "saving.py",
        "sequential.py",
        "training.py",
        "training_arrays_v1.py",
        "training_distributed_v1.py",
        "training_eager_v1.py",
        "training_generator_v1.py",
        "training_utils.py",
        "training_utils_v1.py",
        "training_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        ":base_preprocessing_layer",
        ":data_adapter",
        ":input_spec",
        ":keras_tensor",
        ":node",
        "//tensorflow/python:py_checkpoint_reader",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:callbacks_v1",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:metrics",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/keras/distribute:distribute_coordinator_utils",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/mixed_precision:autocast_variable",
        "//tensorflow/python/keras/mixed_precision:loss_scale_optimizer",
        "//tensorflow/python/keras/mixed_precision:policy",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/keras/utils:version_utils",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_util",
        "//tensorflow/python/profiler:trace",
        "//tensorflow/python/saved_model:constants",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "base_layer_utils",
    srcs = ["base_layer_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:auto_control_deps",
        "//tensorflow/python:control_flow_v2_func_graphs",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:tf_inspect",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "base_layer",
    srcs = [
        "base_layer.py",
        "base_layer_v1.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer_utils",
        ":input_spec",
        ":node",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:auto_control_deps",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras/initializers",
        # TODO(keras-team): Fix the cyclar deps between layer and metrics.
        # "//tensorflow/python/keras:metrics",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/mixed_precision:autocast_variable",
        "//tensorflow/python/keras/mixed_precision:loss_scale_optimizer",
        "//tensorflow/python/keras/mixed_precision:policy",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:object_identity",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/keras/utils:version_utils",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/training/tracking",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/python/training/tracking:layer_utils",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "data_adapter",
    srcs = ["data_adapter.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/utils:dataset_creator",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "input_spec",
    srcs = ["input_spec.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras:backend",
    ],
)

py_library(
    name = "keras_tensor",
    srcs = ["keras_tensor.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras/utils:object_identity",
    ],
)

py_library(
    name = "base_preprocessing_layer",
    srcs = [
        "base_preprocessing_layer.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":base_layer",
        "//tensorflow/python/data",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/module",
    ],
)

py_library(
    name = "node",
    srcs = ["node.py"],
    srcs_version = "PY3",
    deps = [
        ":base_layer_utils",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/utils:tf_utils",
        "//third_party/py/numpy",
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
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:combinations",
    ],
)

tf_py_test(
    name = "data_adapter_test",
    size = "medium",
    srcs = ["data_adapter_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss_py38",  # TODO(b/150615192)
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":data_adapter",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
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
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":engine",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras/layers:convolutional",
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
    name = "keras_tensor_test",
    size = "small",
    srcs = ["keras_tensor_test.py"],
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
    name = "ragged_keras_tensor_test",
    size = "small",
    srcs = ["ragged_keras_tensor_test.py"],
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
        "manual",
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
    deps = [
        ":engine",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:callbacks",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:metrics",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/utils:data_utils",
        "//tensorflow/python/keras/utils:np_utils",
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
        ":engine",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:metrics",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/utils:data_utils",
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
    name = "training_utils_v1_test",
    size = "medium",
    srcs = ["training_utils_v1_test.py"],
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
    name = "input_layer_test",
    size = "medium",
    srcs = ["input_layer_test.py"],
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/utils:layer_utils",
    ],
)

tf_py_test(
    name = "functional_test",
    size = "medium",
    srcs = ["functional_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no-internal-py3",
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:models",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/training/tracking:util",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "node_test",
    size = "medium",
    srcs = ["node_test.py"],
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/utils:layer_utils",
    ],
)

tf_py_test(
    name = "base_layer_test",
    size = "medium",
    srcs = ["base_layer_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":base_layer",
        ":engine",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/legacy_tf_layers:core",
        "//tensorflow/python/keras/mixed_precision:policy",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
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

tf_py_test(
    name = "deferred_sequential_test",
    size = "medium",
    srcs = ["deferred_sequential_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
