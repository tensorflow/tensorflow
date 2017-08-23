# Description:
#   Contains the Keras API (internal TensorFlow version).

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "api/__init__.py",
        "api/keras/__init__.py",
        "api/keras/activations/__init__.py",
        "api/keras/applications/__init__.py",
        "api/keras/applications/inception_v3/__init__.py",
        "api/keras/applications/mobilenet/__init__.py",
        "api/keras/applications/resnet50/__init__.py",
        "api/keras/applications/vgg16/__init__.py",
        "api/keras/applications/vgg19/__init__.py",
        "api/keras/applications/xception/__init__.py",
        "api/keras/backend/__init__.py",
        "api/keras/callbacks/__init__.py",
        "api/keras/constraints/__init__.py",
        "api/keras/datasets/__init__.py",
        "api/keras/datasets/boston_housing/__init__.py",
        "api/keras/datasets/cifar10/__init__.py",
        "api/keras/datasets/cifar100/__init__.py",
        "api/keras/datasets/imdb/__init__.py",
        "api/keras/datasets/mnist/__init__.py",
        "api/keras/datasets/reuters/__init__.py",
        "api/keras/initializers/__init__.py",
        "api/keras/layers/__init__.py",
        "api/keras/losses/__init__.py",
        "api/keras/metrics/__init__.py",
        "api/keras/models/__init__.py",
        "api/keras/optimizers/__init__.py",
        "api/keras/preprocessing/__init__.py",
        "api/keras/preprocessing/image/__init__.py",
        "api/keras/preprocessing/sequence/__init__.py",
        "api/keras/preprocessing/text/__init__.py",
        "api/keras/regularizers/__init__.py",
        "api/keras/utils/__init__.py",
        "api/keras/wrappers/__init__.py",
        "api/keras/wrappers/scikit_learn/__init__.py",
        "python/keras/__init__.py",
        "python/keras/activations.py",
        "python/keras/applications/__init__.py",
        "python/keras/applications/imagenet_utils.py",
        "python/keras/applications/inception_v3.py",
        "python/keras/applications/mobilenet.py",
        "python/keras/applications/resnet50.py",
        "python/keras/applications/vgg16.py",
        "python/keras/applications/vgg19.py",
        "python/keras/applications/xception.py",
        "python/keras/backend.py",
        "python/keras/callbacks.py",
        "python/keras/constraints.py",
        "python/keras/datasets/__init__.py",
        "python/keras/datasets/boston_housing.py",
        "python/keras/datasets/cifar.py",
        "python/keras/datasets/cifar10.py",
        "python/keras/datasets/cifar100.py",
        "python/keras/datasets/imdb.py",
        "python/keras/datasets/mnist.py",
        "python/keras/datasets/reuters.py",
        "python/keras/engine/__init__.py",
        "python/keras/engine/topology.py",
        "python/keras/engine/training.py",
        "python/keras/initializers.py",
        "python/keras/layers/__init__.py",
        "python/keras/layers/advanced_activations.py",
        "python/keras/layers/convolutional.py",
        "python/keras/layers/convolutional_recurrent.py",
        "python/keras/layers/core.py",
        "python/keras/layers/embeddings.py",
        "python/keras/layers/local.py",
        "python/keras/layers/merge.py",
        "python/keras/layers/noise.py",
        "python/keras/layers/normalization.py",
        "python/keras/layers/pooling.py",
        "python/keras/layers/recurrent.py",
        "python/keras/layers/serialization.py",
        "python/keras/layers/wrappers.py",
        "python/keras/losses.py",
        "python/keras/metrics.py",
        "python/keras/models.py",
        "python/keras/optimizers.py",
        "python/keras/preprocessing/__init__.py",
        "python/keras/preprocessing/image.py",
        "python/keras/preprocessing/sequence.py",
        "python/keras/preprocessing/text.py",
        "python/keras/regularizers.py",
        "python/keras/testing_utils.py",
        "python/keras/utils/__init__.py",
        "python/keras/utils/conv_utils.py",
        "python/keras/utils/data_utils.py",
        "python/keras/utils/generic_utils.py",
        "python/keras/utils/io_utils.py",
        "python/keras/utils/layer_utils.py",
        "python/keras/utils/np_utils.py",
        "python/keras/utils/vis_utils.py",
        "python/keras/wrappers/__init__.py",
        "python/keras/wrappers/scikit_learn.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/tensorboard:projector",
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
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
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
        "@six_archive//:six",
    ],
)

py_test(
    name = "integration_test",
    size = "medium",
    srcs = ["python/keras/integration_test.py"],
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
    srcs = ["python/keras/activations_test.py"],
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
    srcs = ["python/keras/constraints_test.py"],
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
    srcs = ["python/keras/initializers_test.py"],
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
    srcs = ["python/keras/regularizers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["python/keras/optimizers_test.py"],
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
    srcs = ["python/keras/losses_test.py"],
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
    srcs = ["python/keras/metrics_test.py"],
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
    srcs = ["python/keras/applications/inception_v3_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "mobilenet_test",
    size = "small",
    srcs = ["python/keras/applications/mobilenet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "resnet50_test",
    size = "small",
    srcs = ["python/keras/applications/resnet50_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "vgg16_test",
    size = "small",
    srcs = ["python/keras/applications/vgg16_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "vgg19_test",
    size = "small",
    srcs = ["python/keras/applications/vgg19_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "xception_test",
    size = "medium",
    srcs = ["python/keras/applications/xception_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "advanced_activations_test",
    size = "small",
    srcs = ["python/keras/layers/advanced_activations_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "convolutional_recurrent_test",
    size = "medium",
    srcs = ["python/keras/layers/convolutional_recurrent_test.py"],
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
    size = "medium",
    srcs = ["python/keras/layers/convolutional_test.py"],
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
    srcs = ["python/keras/layers/pooling_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "core_test",
    size = "small",
    srcs = ["python/keras/layers/core_test.py"],
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
    srcs = ["python/keras/layers/embeddings_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "local_test",
    size = "medium",
    srcs = ["python/keras/layers/local_test.py"],
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
    srcs = ["python/keras/layers/merge_test.py"],
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
    srcs = ["python/keras/layers/noise_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "normalization_test",
    size = "small",
    srcs = ["python/keras/layers/normalization_test.py"],
    additional_deps = [
        ":keras",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "simplernn_test",
    size = "medium",
    srcs = ["python/keras/layers/simplernn_test.py"],
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
    srcs = ["python/keras/layers/gru_test.py"],
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
    srcs = ["python/keras/layers/lstm_test.py"],
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
    name = "serialization_test",
    size = "small",
    srcs = ["python/keras/layers/serialization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "wrappers_test",
    size = "small",
    srcs = ["python/keras/layers/wrappers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "scikit_learn_test",
    size = "small",
    srcs = ["python/keras/wrappers/scikit_learn_test.py"],
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
    size = "small",
    srcs = ["python/keras/utils/data_utils_test.py"],
    srcs_version = "PY2AND3",
    tags = [
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
    srcs = ["python/keras/utils/generic_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "imagenet_utils_test",
    size = "small",
    srcs = ["python/keras/applications/imagenet_utils_test.py"],
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
    srcs = ["python/keras/preprocessing/image_test.py"],
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
    srcs = ["python/keras/preprocessing/sequence_test.py"],
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
    srcs = ["python/keras/preprocessing/text_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "callbacks_test",
    size = "small",
    srcs = ["python/keras/callbacks_test.py"],
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
    size = "small",
    srcs = ["python/keras/engine/training_test.py"],
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
    srcs = ["python/keras/engine/topology_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "models_test",
    size = "small",
    srcs = ["python/keras/models_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "backend_test",
    size = "small",
    srcs = ["python/keras/backend_test.py"],
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
        "python/keras/testing_utils.py",
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
