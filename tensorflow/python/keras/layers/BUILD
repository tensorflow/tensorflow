# Description:
#   Contains the Keras layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/feature_column:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/training/tracking:__pkg__",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

# A separate build for layers without serialization to avoid circular deps
# with feature column.
py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "serialization.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":advanced_activations",
        ":convolutional",
        ":convolutional_recurrent",
        ":core",
        ":cudnn_recurrent",
        ":dense_attention",
        ":einsum_dense",
        ":embeddings",
        ":kernelized",
        ":local",
        ":merge",
        ":noise",
        ":normalization",
        ":normalization_v2",
        ":pooling",
        ":recurrent",
        ":recurrent_v2",
        ":rnn_cell_wrapper_v2",
        ":wrappers",
        "//tensorflow/python/keras/layers/preprocessing",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "advanced_activations",
    srcs = ["advanced_activations.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pooling",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "convolutional_recurrent",
    srcs = ["convolutional_recurrent.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":recurrent",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/training/tracking:base",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cudnn_recurrent",
    srcs = ["cudnn_recurrent.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":recurrent",
        ":recurrent_v2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:cudnn_rnn_ops_gen",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
    ],
)

py_library(
    name = "dense_attention",
    srcs = ["dense_attention.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "einsum_dense",
    srcs = ["einsum_dense.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
    ],
)

py_library(
    name = "embeddings",
    srcs = ["embeddings.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "kernelized",
    srcs = ["kernelized.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras/engine:input_spec",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "local",
    srcs = ["local.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "merge",
    srcs = ["merge.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "noise",
    srcs = ["noise.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/utils:tf_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "normalization",
    srcs = ["normalization.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "normalization_v2",
    srcs = ["normalization_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":normalization",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:initializers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "recurrent_v2",
    srcs = ["recurrent_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":recurrent",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:cudnn_rnn_ops_gen",
        "//tensorflow/python:device",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_build_info",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine:input_spec",
    ],
)

py_library(
    name = "rnn_cell_wrapper_v2",
    srcs = ["rnn_cell_wrapper_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":recurrent",
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "wrappers",
    srcs = ["wrappers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":recurrent",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

tf_py_test(
    name = "advanced_activations_test",
    size = "medium",
    srcs = ["advanced_activations_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "tensorflow_op_layer_test",
    size = "medium",
    srcs = ["tensorflow_op_layer_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/saving",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "convolutional_recurrent_test",
    size = "medium",
    srcs = ["convolutional_recurrent_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = ["no_rocm"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "convolutional_test",
    size = "medium",
    srcs = ["convolutional_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "convolutional_transpose_test",
    size = "medium",
    srcs = ["convolutional_transpose_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "cudnn_recurrent_test",
    size = "medium",
    srcs = ["cudnn_recurrent_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "pooling_test",
    size = "medium",
    srcs = ["pooling_test.py"],
    python_version = "PY3",
    shard_count = 8,
    # TODO(b/127881287): Re-enable.
    tags = [
        "no_windows_gpu",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "core_test",
    size = "medium",
    srcs = ["core_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "subclassed_layers_test",
    size = "medium",
    srcs = ["subclassed_layers_test.py"],
    python_version = "PY3",
    shard_count = 3,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "dense_attention_test",
    size = "medium",
    srcs = ["dense_attention_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "embeddings_test",
    size = "medium",
    srcs = ["embeddings_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
    ],
)

tf_py_test(
    name = "einsum_dense_test",
    srcs = ["einsum_dense_test.py"],
    python_version = "PY3",
    deps = [
        ":einsum_dense",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "local_test",
    size = "medium",
    srcs = ["local_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "merge_test",
    size = "medium",
    srcs = ["merge_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "noise_test",
    size = "small",
    srcs = ["noise_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "normalization_test",
    size = "medium",
    srcs = ["normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "simplernn_test",
    size = "medium",
    srcs = ["simplernn_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["notsan"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "gru_test",
    size = "medium",
    srcs = ["gru_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
        "notsan",  # http://b/62136390
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["lstm_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
        "noasan",  # times out b/63678675
        "notsan",  # http://b/62189182
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "recurrent_test",
    size = "medium",
    srcs = ["recurrent_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = ["no_rocm"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "recurrent_v2_test",
    size = "medium",
    srcs = ["recurrent_v2_test.py"],
    python_version = "PY3",
    shard_count = 2,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "separable_convolutional_test",
    size = "medium",
    srcs = ["separable_convolutional_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "lstm_v2_test",
    size = "medium",
    srcs = ["lstm_v2_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "gru_v2_test",
    size = "medium",
    srcs = ["gru_v2_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["serialization_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "kernelized_test",
    size = "small",
    srcs = ["kernelized_test.py"],
    python_version = "PY3",
    deps = [
        ":layers",
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
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:initializers",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "wrappers_test",
    size = "medium",
    srcs = ["wrappers_test.py"],
    python_version = "PY3",
    shard_count = 12,
    tags = [
        "noasan",  # http://b/78599823
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/ops/ragged:ragged_concat_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "rnn_cell_wrapper_v2_test",
    size = "medium",
    srcs = ["rnn_cell_wrapper_v2_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
