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
        "preprocessing/__init__.py",
        "preprocessing/image.py",
        "preprocessing/sequence.py",
        "preprocessing/text.py",
        "testing_utils.py",
        "utils/__init__.py",
        "utils/all_utils.py",
        "utils/multi_gpu_utils.py",
        "utils/np_utils.py",
        "utils/vis_utils.py",
        "wrappers/__init__.py",
        "wrappers/scikit_learn.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":backend",
        ":engine",
        ":layers",
        ":saving",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras/applications",
        "//tensorflow/python/keras/datasets",
        "//tensorflow/python/keras/mixed_precision/experimental:mixed_precision_experimental",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/premade",
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
        ":tf_utils",
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
        "utils/metrics_utils.py",
        "utils/version_utils.py",
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
        ":engine_utils",
        ":initializers",
        ":input_spec",
        ":losses",
        ":mode_keys",
        ":optimizers",
        ":regularizers",
        ":saving",
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
        ":engine_utils",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
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
        ":engine_utils",
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
    name = "saving",
    srcs = [
        "saving/__init__.py",
        "saving/hdf5_format.py",
        "saving/model_config.py",
        "saving/save.py",
        "saving/saved_model/base_serialization.py",
        "saving/saved_model/constants.py",
        "saving/saved_model/layer_serialization.py",
        "saving/saved_model/load.py",
        "saving/saved_model/model_serialization.py",
        "saving/saved_model/network_serialization.py",
        "saving/saved_model/save.py",
        "saving/saved_model/save_impl.py",
        "saving/saved_model/serialized_attributes.py",
        "saving/saved_model/utils.py",
        "saving/saved_model_experimental.py",
        "saving/saving_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":engine_utils",
        ":input_spec",
        ":mode_keys",
        ":optimizers",
        ":regularizers",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:saver",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model/model_utils",
        "//tensorflow/python/training/tracking",
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
        ":engine_utils",
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
        ":engine_utils",
        ":mode_keys",
        "//tensorflow/python/distribute:distributed_file_utils",
        "//tensorflow/python/keras/distribute:multi_worker_training_state",
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
        ":engine_utils",
        "//tensorflow/python/eager:profiler",
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
        ":engine_utils",
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
        ":engine_utils",
        "//tensorflow/python:init_ops_v2",
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
        ":engine_utils",
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
        ":engine_utils",
        "//tensorflow/python/keras/optimizer_v2",
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
        ":engine_utils",
    ],
)

py_library(
    name = "engine_utils",
    srcs = [
        "utils/conv_utils.py",
        "utils/data_utils.py",
        "utils/io_utils.py",
        "utils/losses_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/ops/losses:loss_reduction",
    ],
)

py_library(
    name = "tf_utils",
    srcs = ["utils/tf_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:smart_cond",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "@six_archive//:six",
    ],
)

# A separate build for layers without serialization to avoid circular deps
# with feature column.
py_library(
    name = "layers_base",
    srcs = [
        "layers/__init__.py",
        "layers/advanced_activations.py",
        "layers/convolutional.py",
        "layers/convolutional_recurrent.py",
        "layers/core.py",
        "layers/cudnn_recurrent.py",
        "layers/dense_attention.py",
        "layers/embeddings.py",
        "layers/kernelized.py",
        "layers/local.py",
        "layers/merge.py",
        "layers/noise.py",
        "layers/normalization.py",
        "layers/normalization_v2.py",
        "layers/pooling.py",
        "layers/preprocessing/image_preprocessing.py",
        "layers/preprocessing/normalization.py",
        "layers/preprocessing/normalization_v1.py",
        "layers/preprocessing/text_vectorization.py",
        "layers/preprocessing/text_vectorization_v1.py",
        "layers/recurrent.py",
        "layers/recurrent_v2.py",
        "layers/rnn_cell_wrapper_v2.py",
        "layers/wrappers.py",
        "utils/kernelized_utils.py",
        "utils/layer_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        ":generic_utils",
        ":tf_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cudnn_rnn_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "preprocessing_test_utils",
    srcs = ["layers/preprocessing/preprocessing_test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "layers",
    srcs = [
        "layers/serialization.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_base",
        ":tf_utils",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
)

py_library(
    name = "generic_utils",
    srcs = [
        "utils/generic_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "mode_keys",
    srcs = [
        "utils/mode_keys.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/saved_model/model_utils:mode_keys",
    ],
)

tf_py_test(
    name = "integration_test",
    size = "medium",
    srcs = ["integration_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
    ],
    shard_count = 16,
    tags = ["notsan"],
)

tf_py_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
    ],
)

tf_py_test(
    name = "constraints_test",
    size = "small",
    srcs = ["constraints_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "data_adapter_test",
    size = "medium",
    srcs = ["engine/data_adapter_test.py"],
    additional_deps = [
        ":data_adapter",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["initializers_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:init_ops",
    ],
)

tf_py_test(
    name = "regularizers_test",
    size = "medium",
    srcs = ["regularizers_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["optimizers_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
    ],
    shard_count = 8,
    tags = ["notsan"],
)

tf_py_test(
    name = "losses_test",
    size = "small",
    srcs = ["losses_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "metrics_functional_test",
    size = "small",
    srcs = ["metrics_functional_test.py"],
    additional_deps = [
        ":keras",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "metrics_confusion_matrix_test",
    size = "medium",
    srcs = ["metrics_confusion_matrix_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "metrics_correctness_test",
    size = "medium",
    srcs = ["metrics_correctness_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "temporal_sample_weights_correctness_test",
    size = "medium",
    srcs = ["temporal_sample_weights_correctness_test.py"],
    additional_deps = [
        ":keras",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 20,
)

tf_py_test(
    name = "advanced_activations_test",
    size = "medium",
    srcs = ["layers/advanced_activations_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tensorflow_op_layer_test",
    size = "medium",
    srcs = ["layers/tensorflow_op_layer_test.py"],
    additional_deps = [
        ":keras",
        ":saving",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
    ],
    shard_count = 3,
)

tf_py_test(
    name = "convolutional_recurrent_test",
    size = "medium",
    srcs = ["layers/convolutional_recurrent_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = ["no_rocm"],
)

cuda_py_test(
    name = "convolutional_test",
    size = "medium",
    srcs = ["layers/convolutional_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 8,
)

cuda_py_test(
    name = "image_preprocessing_test",
    size = "medium",
    srcs = ["layers/preprocessing/image_preprocessing_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
)

cuda_py_test(
    name = "convolutional_transpose_test",
    size = "medium",
    srcs = ["layers/convolutional_transpose_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "cudnn_recurrent_test",
    size = "medium",
    srcs = ["layers/cudnn_recurrent_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows_gpu",
    ],
)

tf_py_test(
    name = "base_preprocessing_layer_test",
    size = "medium",
    srcs = ["engine/base_preprocessing_layer_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "pooling_test",
    size = "medium",
    srcs = ["layers/pooling_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 8,
    # TODO(b/127881287): Re-enable.
    tags = [
        "no_windows_gpu",
    ],
)

tf_py_test(
    name = "core_test",
    size = "medium",
    srcs = ["layers/core_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 3,
)

tf_py_test(
    name = "subclassed_layers_test",
    size = "medium",
    srcs = ["layers/subclassed_layers_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 3,
)

tf_py_test(
    name = "dense_attention_test",
    size = "medium",
    srcs = ["layers/dense_attention_test.py"],
    additional_deps = [
        ":keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "embeddings_test",
    size = "medium",
    srcs = ["layers/embeddings_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "local_test",
    size = "medium",
    srcs = ["layers/local_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = ["no_windows"],
)

tf_py_test(
    name = "merge_test",
    size = "medium",
    srcs = ["layers/merge_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "noise_test",
    size = "small",
    srcs = ["layers/noise_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "normalization_test",
    size = "medium",
    srcs = ["layers/normalization_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "no_rocm",
        "notsan",
    ],
)

tf_py_test(
    name = "preprocessing_normalization_test",
    size = "small",
    srcs = ["layers/preprocessing/normalization_test.py"],
    additional_deps = [
        ":keras",
        ":preprocessing_test_utils",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    main = "normalization_test.py",
)

tf_py_test(
    name = "preprocessing_text_vectorization_test",
    size = "medium",
    srcs = ["layers/preprocessing/text_vectorization_test.py"],
    additional_deps = [
        ":generic_utils",
        ":keras",
        ":preprocessing_test_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    main = "text_vectorization_test.py",
)

tf_py_test(
    name = "simplernn_test",
    size = "medium",
    srcs = ["layers/simplernn_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = ["notsan"],
)

tf_py_test(
    name = "gru_test",
    size = "medium",
    srcs = ["layers/gru_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = ["notsan"],  # http://b/62136390
)

tf_py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["layers/lstm_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "noasan",  # times out b/63678675
        "notsan",  # http://b/62189182
    ],
)

tf_py_test(
    name = "recurrent_test",
    size = "medium",
    srcs = ["layers/recurrent_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 10,
)

cuda_py_test(
    name = "recurrent_v2_test",
    size = "medium",
    srcs = ["layers/recurrent_v2_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 2,
)

cuda_py_test(
    name = "separable_convolutional_test",
    size = "medium",
    srcs = ["layers/separable_convolutional_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "lstm_v2_test",
    size = "medium",
    srcs = ["layers/lstm_v2_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 12,
    tags = ["no_rocm"],
)

cuda_py_test(
    name = "gru_v2_test",
    size = "medium",
    srcs = ["layers/gru_v2_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 12,
    tags = ["no_rocm"],
)

tf_py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["layers/serialization_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "kernelized_test",
    size = "small",
    srcs = ["layers/kernelized_test.py"],
    additional_deps = [
        ":backend",
        ":initializers",
        ":keras",
        ":layers",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/eager:context",
    ],
)

tf_py_test(
    name = "wrappers_test",
    size = "large",
    srcs = ["layers/wrappers_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_concat_ops",
    ],
    shard_count = 6,
    tags = [
        "noasan",  # http://b/78599823
        "notsan",
    ],
)

tf_py_test(
    name = "rnn_cell_wrapper_v2_test",
    size = "medium",
    srcs = ["layers/rnn_cell_wrapper_v2_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "notsan",
    ],
)

tf_py_test(
    name = "time_distributed_learning_phase_test",
    size = "small",
    srcs = ["layers/time_distributed_learning_phase_test.py"],
    additional_deps = [
        ":keras",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "noasan",  # http://b/78599823
        "notsan",
    ],
)

tf_py_test(
    name = "scikit_learn_test",
    size = "small",
    srcs = ["wrappers/scikit_learn_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = ["notsan"],
)

tf_py_test(
    name = "data_utils_test",
    size = "medium",
    srcs = ["utils/data_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # times out
        "notsan",
        "optonly",  # times out
    ],
)

tf_py_test(
    name = "generic_utils_test",
    size = "small",
    srcs = ["utils/generic_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "version_utils_test",
    size = "small",
    srcs = ["utils/version_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "tf_utils_test",
    size = "small",
    srcs = ["utils/tf_utils_test.py"],
    additional_deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "composite_tensor_support_test",
    size = "medium",
    srcs = ["utils/composite_tensor_support_test.py"],
    additional_deps = [
        ":engine",
        ":layers",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
    ],
    shard_count = 8,
    tags = ["no_windows"],  # b/135752236
)

tf_py_test(
    name = "io_utils_test",
    size = "small",
    srcs = ["utils/io_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_windows",  # TODO: needs investigation on Windows
        "notsan",
    ],
)

tf_py_test(
    name = "np_utils_test",
    size = "small",
    srcs = ["utils/np_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "kernelized_utils_test",
    size = "small",
    srcs = ["utils/kernelized_utils_test.py"],
    additional_deps = [
        ":layers",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
    ],
)

cuda_py_test(
    name = "multi_gpu_utils_test",
    srcs = ["utils/multi_gpu_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "guitar",
        "multi_gpu",
    ],
    xla_enable_strict_auto_jit = False,  # b/142744009
)

cuda_py_test(
    name = "training_gpu_test",
    size = "small",
    srcs = ["engine/training_gpu_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "vis_utils_test",
    size = "small",
    srcs = ["utils/vis_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "conv_utils_test",
    size = "small",
    srcs = ["utils/conv_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "image_test",
    size = "medium",
    srcs = ["preprocessing/image_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "sequence_test",
    size = "small",
    srcs = ["preprocessing/sequence_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "text_test",
    size = "small",
    srcs = ["preprocessing/text_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "callbacks_test",
    size = "medium",
    srcs = ["callbacks_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python:client_testlib",
    ],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",
        "notsan",
    ],
)

tf_py_test(
    name = "callbacks_v1_test",
    size = "medium",
    srcs = ["callbacks_v1_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = ["notsan"],
)

tf_py_test(
    name = "correctness_test",
    size = "medium",
    srcs = ["engine/correctness_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 2,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
)

tf_py_test(
    name = "input_spec_test",
    size = "small",
    srcs = ["engine/input_spec_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "training_test",
    size = "medium",
    srcs = ["engine/training_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 20,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
)

tf_py_test(
    name = "compile_utils_test",
    size = "medium",
    srcs = ["engine/compile_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "training_dataset_test",
    size = "medium",
    srcs = ["engine/training_dataset_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "training_arrays_test",
    size = "medium",
    srcs = ["engine/training_arrays_test.py"],
    additional_deps = [
        ":keras",
        ":layers",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "training_generator_test",
    size = "medium",
    srcs = ["engine/training_generator_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 6,
    tags = [
        "noasan",  # TODO(b/132183295): Re-enable this.
        "nomac",  # TODO(b/140193633): Re-enable this.
        "notsan",
    ],
)

tf_py_test(
    name = "training_integration_test",
    size = "medium",
    srcs = ["engine/training_integration_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 30,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "feature_columns_integration_test",
    size = "medium",
    srcs = ["engine/feature_columns_integration_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
)

tf_py_test(
    name = "training_eager_test",
    size = "medium",
    srcs = ["engine/training_eager_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
        "notsan",
    ],
)

tf_py_test(
    name = "training_utils_test",
    size = "medium",
    srcs = ["engine/training_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
    ],
)

tf_py_test(
    name = "training_v2_utils_test",
    size = "medium",
    srcs = ["engine/training_v2_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:strategy_combinations",
    ],
    tags = [
        "no_oss",  # TODO(b/135021748) reenable
        "notsan",
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
    additional_deps = [
        ":model_subclassing_test_util",
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",
        "notsan",
    ],
)

tf_py_test(
    name = "model_subclassing_compiled_test",
    size = "medium",
    srcs = ["model_subclassing_compiled_test.py"],
    additional_deps = [
        ":model_subclassing_test_util",
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "no_windows",
        "notsan",
    ],
)

tf_py_test(
    name = "custom_training_loop_test",
    size = "medium",
    srcs = ["custom_training_loop_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = ["notsan"],
)

tf_py_test(
    name = "network_test",
    size = "medium",
    srcs = ["engine/network_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 8,
    tags = [
        "no-internal-py3",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "base_layer_test",
    size = "medium",
    srcs = ["engine/base_layer_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "base_layer_utils_test",
    srcs = ["engine/base_layer_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "control_flow_test",
    size = "medium",
    srcs = ["engine/control_flow_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 8,
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "hdf5_format_test",
    size = "medium",
    srcs = ["saving/hdf5_format_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "no_windows",
    ],
)

tf_py_test(
    name = "sequential_test",
    size = "medium",
    srcs = ["engine/sequential_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
)

tf_py_test(
    name = "models_test",
    size = "medium",
    srcs = ["models_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
    ],
    shard_count = 8,
    tags = ["notsan"],  # b/67509773
)

tf_py_test(
    name = "backend_test",
    size = "medium",
    srcs = ["backend_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
    ],
    shard_count = 4,
)

tf_py_test(
    name = "backend_config_test",
    size = "medium",
    srcs = ["backend_config_test.py"],
    additional_deps = [
        ":keras",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "keras_parameterized_test",
    size = "small",
    srcs = ["keras_parameterized_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = ["notsan"],
)

tf_py_test(
    name = "save_test",
    size = "medium",
    srcs = ["saving/save_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/feature_column:feature_column_v2",
    ],
)

tf_py_test(
    name = "saved_model_experimental_test",
    size = "medium",
    srcs = ["saving/saved_model_experimental_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "no_oss",  # TODO(b/119349471): Re-enable
        "no_windows",
    ],
)

tf_py_test(
    name = "saved_model_test",
    size = "medium",
    srcs = ["saving/saved_model/saved_model_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
    tags = [
        "no_windows",
    ],
)

tf_py_test(
    name = "saving_utils_test",
    size = "medium",
    srcs = ["saving/saving_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = ["notsan"],
)

tf_py_test(
    name = "metrics_utils_test",
    size = "small",
    srcs = ["utils/metrics_utils_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
    ],
)
