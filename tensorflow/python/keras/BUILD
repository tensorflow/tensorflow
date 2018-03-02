# Description:
#   Contains the Keras API (internal TensorFlow version).

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "_impl/keras/__init__.py",
        "_impl/keras/activations.py",
        "_impl/keras/applications/__init__.py",
        "_impl/keras/applications/densenet.py",
        "_impl/keras/applications/imagenet_utils.py",
        "_impl/keras/applications/inception_resnet_v2.py",
        "_impl/keras/applications/inception_v3.py",
        "_impl/keras/applications/mobilenet.py",
        "_impl/keras/applications/nasnet.py",
        "_impl/keras/applications/resnet50.py",
        "_impl/keras/applications/vgg16.py",
        "_impl/keras/applications/vgg19.py",
        "_impl/keras/applications/xception.py",
        "_impl/keras/backend.py",
        "_impl/keras/callbacks.py",
        "_impl/keras/constraints.py",
        "_impl/keras/datasets/__init__.py",
        "_impl/keras/datasets/boston_housing.py",
        "_impl/keras/datasets/cifar.py",
        "_impl/keras/datasets/cifar10.py",
        "_impl/keras/datasets/cifar100.py",
        "_impl/keras/datasets/fashion_mnist.py",
        "_impl/keras/datasets/imdb.py",
        "_impl/keras/datasets/mnist.py",
        "_impl/keras/datasets/reuters.py",
        "_impl/keras/engine/__init__.py",
        "_impl/keras/engine/base_layer.py",
        "_impl/keras/engine/input_layer.py",
        "_impl/keras/engine/network.py",
        "_impl/keras/engine/saving.py",
        "_impl/keras/engine/sequential.py",
        "_impl/keras/engine/training.py",
        "_impl/keras/engine/training_eager.py",
        "_impl/keras/estimator.py",
        "_impl/keras/initializers.py",
        "_impl/keras/layers/__init__.py",
        "_impl/keras/layers/advanced_activations.py",
        "_impl/keras/layers/convolutional.py",
        "_impl/keras/layers/convolutional_recurrent.py",
        "_impl/keras/layers/core.py",
        "_impl/keras/layers/embeddings.py",
        "_impl/keras/layers/local.py",
        "_impl/keras/layers/merge.py",
        "_impl/keras/layers/noise.py",
        "_impl/keras/layers/normalization.py",
        "_impl/keras/layers/pooling.py",
        "_impl/keras/layers/recurrent.py",
        "_impl/keras/layers/serialization.py",
        "_impl/keras/layers/wrappers.py",
        "_impl/keras/losses.py",
        "_impl/keras/metrics.py",
        "_impl/keras/models.py",
        "_impl/keras/optimizers.py",
        "_impl/keras/preprocessing/__init__.py",
        "_impl/keras/preprocessing/image.py",
        "_impl/keras/preprocessing/sequence.py",
        "_impl/keras/preprocessing/text.py",
        "_impl/keras/regularizers.py",
        "_impl/keras/testing_utils.py",
        "_impl/keras/utils/__init__.py",
        "_impl/keras/utils/conv_utils.py",
        "_impl/keras/utils/data_utils.py",
        "_impl/keras/utils/generic_utils.py",
        "_impl/keras/utils/io_utils.py",
        "_impl/keras/utils/layer_utils.py",
        "_impl/keras/utils/np_utils.py",
        "_impl/keras/utils/training_utils.py",
        "_impl/keras/utils/vis_utils.py",
        "_impl/keras/wrappers/__init__.py",
        "_impl/keras/wrappers/scikit_learn.py",
        "activations/__init__.py",
        "applications/__init__.py",
        "applications/densenet/__init__.py",
        "applications/inception_resnet_v2/__init__.py",
        "applications/inception_v3/__init__.py",
        "applications/mobilenet/__init__.py",
        "applications/nasnet/__init__.py",
        "applications/resnet50/__init__.py",
        "applications/vgg16/__init__.py",
        "applications/vgg19/__init__.py",
        "applications/xception/__init__.py",
        "backend/__init__.py",
        "callbacks/__init__.py",
        "constraints/__init__.py",
        "datasets/__init__.py",
        "datasets/boston_housing/__init__.py",
        "datasets/cifar10/__init__.py",
        "datasets/cifar100/__init__.py",
        "datasets/fashion_mnist/__init__.py",
        "datasets/imdb/__init__.py",
        "datasets/mnist/__init__.py",
        "datasets/reuters/__init__.py",
        "estimator/__init__.py",
        "initializers/__init__.py",
        "layers/__init__.py",
        "losses/__init__.py",
        "metrics/__init__.py",
        "models/__init__.py",
        "optimizers/__init__.py",
        "preprocessing/__init__.py",
        "preprocessing/image/__init__.py",
        "preprocessing/sequence/__init__.py",
        "preprocessing/text/__init__.py",
        "regularizers/__init__.py",
        "utils/__init__.py",
        "wrappers/__init__.py",
        "wrappers/scikit_learn/__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:ctc_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:logging_ops",
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
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator",
        "//tensorflow/python/estimator:model_fn",
        "//tensorflow/python/saved_model",
        "@six_archive//:six",
    ],
)

py_test(
    name = "integration_test",
    size = "medium",
    srcs = ["_impl/keras/integration_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:layers",
        "//tensorflow/python:nn",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "activations_test",
    size = "small",
    srcs = ["_impl/keras/activations_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "constraints_test",
    size = "small",
    srcs = ["_impl/keras/constraints_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["_impl/keras/initializers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:init_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "regularizers_test",
    size = "small",
    srcs = ["_impl/keras/regularizers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["_impl/keras/optimizers_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "losses_test",
    size = "small",
    srcs = ["_impl/keras/losses_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "metrics_test",
    size = "small",
    srcs = ["_impl/keras/metrics_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "densenet_test",
    size = "large",
    srcs = ["_impl/keras/applications/densenet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "inception_resnet_v2_test",
    size = "medium",
    srcs = ["_impl/keras/applications/inception_resnet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "inception_v3_test",
    size = "medium",
    srcs = ["_impl/keras/applications/inception_v3_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "mobilenet_test",
    size = "medium",
    srcs = ["_impl/keras/applications/mobilenet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "nasnet_test",
    size = "large",
    srcs = ["_impl/keras/applications/nasnet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "resnet50_test",
    size = "small",
    srcs = ["_impl/keras/applications/resnet50_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "vgg16_test",
    size = "small",
    srcs = ["_impl/keras/applications/vgg16_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "vgg19_test",
    size = "small",
    srcs = ["_impl/keras/applications/vgg19_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "xception_test",
    size = "medium",
    srcs = ["_impl/keras/applications/xception_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "advanced_activations_test",
    size = "small",
    srcs = ["_impl/keras/layers/advanced_activations_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "convolutional_recurrent_test",
    size = "medium",
    srcs = ["_impl/keras/layers/convolutional_recurrent_test.py"],
    shard_count = 2,
    srcs_version = "PY2AND3",
    tags = ["noasan"],  # times out b/63678675
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "convolutional_test",
    size = "large",
    srcs = ["_impl/keras/layers/convolutional_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "noasan",  # times out b/63678675
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "pooling_test",
    size = "small",
    srcs = ["_impl/keras/layers/pooling_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "core_test",
    size = "medium",
    srcs = ["_impl/keras/layers/core_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "embeddings_test",
    size = "small",
    srcs = ["_impl/keras/layers/embeddings_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "local_test",
    size = "medium",
    srcs = ["_impl/keras/layers/local_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "merge_test",
    size = "small",
    srcs = ["_impl/keras/layers/merge_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "noise_test",
    size = "small",
    srcs = ["_impl/keras/layers/noise_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "normalization_test",
    size = "small",
    srcs = ["_impl/keras/layers/normalization_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "simplernn_test",
    size = "medium",
    srcs = ["_impl/keras/layers/simplernn_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "gru_test",
    size = "medium",
    srcs = ["_impl/keras/layers/gru_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # http://b/62136390
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "lstm_test",
    size = "medium",
    srcs = ["_impl/keras/layers/lstm_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "noasan",  # times out b/63678675
        "notsan",  # http://b/62189182
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "recurrent_test",
    size = "medium",
    srcs = ["_impl/keras/layers/recurrent_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "serialization_test",
    size = "small",
    srcs = ["_impl/keras/layers/serialization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "wrappers_test",
    size = "medium",
    srcs = ["_impl/keras/layers/wrappers_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "scikit_learn_test",
    size = "small",
    srcs = ["_impl/keras/wrappers/scikit_learn_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "data_utils_test",
    size = "medium",
    srcs = ["_impl/keras/utils/data_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "noasan",  # times out
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "generic_utils_test",
    size = "small",
    srcs = ["_impl/keras/utils/generic_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "io_utils_test",
    size = "small",
    srcs = ["_impl/keras/utils/io_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "np_utils_test",
    size = "small",
    srcs = ["_impl/keras/utils/np_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "training_utils_test",
    size = "medium",
    srcs = ["_impl/keras/utils/training_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = ["multi_gpu"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "imagenet_utils_test",
    size = "small",
    srcs = ["_impl/keras/applications/imagenet_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "image_test",
    size = "medium",
    srcs = ["_impl/keras/preprocessing/image_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sequence_test",
    size = "small",
    srcs = ["_impl/keras/preprocessing/sequence_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "text_test",
    size = "small",
    srcs = ["_impl/keras/preprocessing/text_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "callbacks_test",
    size = "medium",
    srcs = ["_impl/keras/callbacks_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "training_test",
    size = "medium",
    srcs = ["_impl/keras/engine/training_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "training_eager_test",
    size = "medium",
    srcs = ["_impl/keras/engine/training_eager_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "model_subclassing_test",
    size = "medium",
    srcs = ["_impl/keras/model_subclassing_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "topology_test",
    size = "small",
    srcs = ["_impl/keras/engine/topology_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "saving_test",
    size = "small",
    srcs = ["_impl/keras/engine/saving_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sequential_test",
    size = "small",
    srcs = ["_impl/keras/engine/sequential_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "models_test",
    size = "small",
    srcs = ["_impl/keras/models_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],  # b/67509773
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "estimator_test",
    size = "large",
    srcs = ["_impl/keras/estimator_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python/estimator:numpy_io",
        "//tensorflow/python/estimator:run_config",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "backend_test",
    size = "small",
    srcs = ["_impl/keras/backend_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "testing_utils",
    srcs = [
        "_impl/keras/testing_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
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
    visibility = ["//tensorflow:__subpackages__"],
)
