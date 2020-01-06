# Description:
#   Contains the Keras Application package (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "applications",
    srcs = [
        "__init__.py",
        "densenet.py",
        "efficientnet.py",
        "imagenet_utils.py",
        "inception_resnet_v2.py",
        "inception_v3.py",
        "mobilenet.py",
        "mobilenet_v2.py",
        "nasnet.py",
        "resnet.py",
        "resnet_v2.py",
        "vgg16.py",
        "vgg19.py",
        "xception.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:layers_base",
    ],
)

tf_py_test(
    name = "applications_test",
    size = "medium",
    srcs = ["applications_test.py"],
    shard_count = 32,
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "imagenet_utils_test",
    size = "medium",
    srcs = ["imagenet_utils_test.py"],
    shard_count = 2,
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)
