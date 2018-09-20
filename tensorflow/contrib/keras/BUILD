# Description:
#   Contains the Keras API (internal TensorFlow version).
#   Note that tf.contrib.keras has been deprecated in favor of tf.keras.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

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
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/keras",
    ],
)
