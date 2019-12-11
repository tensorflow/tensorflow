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
        "@keras_applications_archive//:keras_applications",
    ],
)

tf_py_test(
    name = "applications_test",
    size = "medium",
    srcs = ["applications_test.py"],
    additional_deps = [
        ":applications",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
    ],
    shard_count = 11,
)
