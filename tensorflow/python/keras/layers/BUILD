# Description:
#   Contains the Keras layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

# A separate build for layers without serialization to avoid circular deps
# with feature column.
py_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "advanced_activations.py",
        "convolutional.py",
        "convolutional_recurrent.py",
        "core.py",
        "cudnn_recurrent.py",
        "dense_attention.py",
        "embeddings.py",
        "kernelized.py",
        "local.py",
        "merge.py",
        "noise.py",
        "normalization.py",
        "normalization_v2.py",
        "pooling.py",
        "recurrent.py",
        "recurrent_v2.py",
        "rnn_cell_wrapper_v2.py",
        "wrappers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
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
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras/layers/preprocessing",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "layers",
    srcs = [
        "serialization.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":layers_base",
        "//tensorflow/python/feature_column:feature_column_py",
        "//tensorflow/python/keras/utils:tf_utils",
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
    shard_count = 4,
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
    name = "simplernn_test",
    size = "medium",
    srcs = ["simplernn_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["notsan"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
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
    tags = ["notsan"],  # http://b/62136390
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
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
    shard_count = 10,
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
    tags = ["no_rocm"],
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
    tags = ["no_rocm"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
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
        "//tensorflow/python/keras:initializers",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "wrappers_test",
    size = "large",
    srcs = ["wrappers_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # http://b/78599823
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
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
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
