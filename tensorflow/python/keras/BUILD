# Description:
#   Contains the Keras API (internal TensorFlow version).
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

config_setting(
    name = "empty_condition",
    values = {"define": "UNUSED=unused"},
)

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "applications/__init__.py",
        "applications/densenet.py",
        "applications/imagenet_utils.py",
        "applications/inception_resnet_v2.py",
        "applications/inception_v3.py",
        "applications/mobilenet.py",
        "applications/mobilenet_v2.py",
        "applications/nasnet.py",
        "applications/resnet50.py",
        "applications/vgg16.py",
        "applications/vgg19.py",
        "applications/xception.py",
        "datasets/__init__.py",
        "datasets/boston_housing.py",
        "datasets/cifar.py",
        "datasets/cifar10.py",
        "datasets/cifar100.py",
        "datasets/fashion_mnist.py",
        "datasets/imdb.py",
        "datasets/mnist.py",
        "datasets/reuters.py",
        "estimator/__init__.py",
        "keras_parameterized.py",
        "ops.py",
        "preprocessing/__init__.py",
        "preprocessing/image.py",
        "preprocessing/sequence.py",
        "preprocessing/text.py",
        "testing_utils.py",
        "utils/__init__.py",
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
        ":pil_for_keras",
        ":saving",
        "//tensorflow/python:training",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/saved_model",
        "@keras_applications_archive//:keras_applications",
    ],
)

py_library(
    name = "pil_for_keras",
    deps = select({
        ":empty_condition": [],
        "//conditions:default": [],
    }),
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
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:ctc_ops",
        "//tensorflow/python:distribute",
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
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_coordinator",
    ],
)

py_library(
    name = "backend_config",
    srcs = ["backend_config.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "engine",
    srcs = [
        "engine/__init__.py",
        "engine/base_layer.py",
        "engine/base_layer_utils.py",
        "engine/distributed_training_utils.py",
        "engine/input_layer.py",
        "engine/input_spec.py",
        "engine/network.py",
        "engine/partial_batch_padding_handler.py",
        "engine/saving.py",
        "engine/sequential.py",
        "engine/training.py",
        "engine/training_arrays.py",
        "engine/training_distributed.py",
        "engine/training_eager.py",
        "engine/training_generator.py",
        "engine/training_utils.py",
        "metrics.py",  # Need base_layer
        "models.py",
        "utils/metrics_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":activations",
        ":backend",
        ":callbacks",
        ":constraints",
        ":engine_utils",
        ":initializers",
        ":losses",
        ":optimizers",
        ":regularizers",
        ":saving",
        "//tensorflow/python/data",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/training/checkpointable:data_structures",
        "//tensorflow/tools/docs:doc_controls",
        "@six_archive//:six",
    ],
)

py_library(
    name = "saving",
    srcs = [
        "saving/__init__.py",
        "saving/hdf5_format.py",
        "saving/model_config.py",
        "saving/saved_model.py",
        "saving/saving_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":engine_utils",
        ":optimizers",
        "//tensorflow/python:lib",
        "//tensorflow/python:mode_keys",
        "//tensorflow/python:saver",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/saved_model/model_utils",
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
    ],
)

py_library(
    name = "layers",
    srcs = [
        "layers/__init__.py",
        "layers/advanced_activations.py",
        "layers/convolutional.py",
        "layers/convolutional_recurrent.py",
        "layers/core.py",
        "layers/cudnn_recurrent.py",
        "layers/embeddings.py",
        "layers/kernelized.py",
        "layers/local.py",
        "layers/merge.py",
        "layers/noise.py",
        "layers/normalization.py",
        "layers/pooling.py",
        "layers/recurrent.py",
        "layers/serialization.py",
        "layers/wrappers.py",
        "utils/kernelized_utils.py",
        "utils/layer_utils.py",
        "utils/tf_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":engine",
        ":generic_utils",
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
    shard_count = 12,
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
    size = "small",
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
    name = "applications_test",
    size = "enormous",
    srcs = ["applications/applications_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 2,
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
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 3,
)

tf_py_test(
    name = "convolutional_recurrent_test",
    size = "large",
    srcs = ["layers/convolutional_recurrent_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 2,
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
    tags = ["no_windows_gpu"],
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
    shard_count = 2,
    tags = ["no_windows"],
)

tf_py_test(
    name = "merge_test",
    size = "small",
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

tf_py_test(
    name = "normalization_test",
    size = "medium",
    srcs = ["layers/normalization_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 3,
    tags = ["notsan"],
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
    shard_count = 4,
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
    name = "unified_lstm_test",
    size = "medium",
    srcs = ["layers/unified_lstm_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 8,
)

cuda_py_test(
    name = "unified_gru_test",
    size = "medium",
    srcs = ["layers/unified_gru_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 8,
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
    size = "medium",
    srcs = ["layers/wrappers_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 4,
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
    shard_count = 6,
    tags = [
        "no_oss",
        "no_windows",
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
    name = "tf_utils_test",
    size = "small",
    srcs = ["utils/tf_utils_test.py"],
    additional_deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
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
    tags = ["notsan"],
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
    shard_count = 16,
    tags = ["notsan"],
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
        "no_oss",
        "notap",  #TODO(b/123544294): Re-enable this test.
        "notsan",
    ],
)

tf_py_test(
    name = "feature_columns_integration_test",
    size = "small",
    srcs = ["engine/feature_columns_integration_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/feature_column:feature_column_py",
    ],
    tags = ["notsan"],
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
    tags = ["notsan"],
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
    tags = ["notsan"],
)

tf_py_test(
    name = "model_subclassing_test",
    size = "medium",
    srcs = ["model_subclassing_test.py"],
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
    name = "topology_test",
    size = "medium",
    srcs = ["engine/topology_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no-internal-py3",
    ],
)

tf_py_test(
    name = "base_layer_test",
    size = "small",
    srcs = ["engine/base_layer_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
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
    name = "saved_model_test",
    size = "medium",
    srcs = ["saving/saved_model_test.py"],
    additional_deps = [
        ":keras",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
    tags = [
        "no_oss",  # TODO(b/119349471): Re-enable
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
