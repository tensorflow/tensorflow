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
        "//tensorflow/python/keras/layers:layers_base",
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

# Add target for each application module file, to make sure it only
# runs the test for the application models contained in that
# application module when it has been modified.
tf_py_test(
    name = "applications_load_weight_test_resnet",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=resnet"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_resnet_v2",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=resnet_v2"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_vgg16",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=vgg16"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_vgg19",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=vgg19"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_xception",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=xception"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_inception_v3",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=inception_v3"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_inception_resnet_v2",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=inception_resnet_v2"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_mobilenet",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=mobilenet"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_mobilenet_v2",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=mobilenet_v2"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_densenet",
    size = "large",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=densenet"],
    main = "applications_load_weight_test.py",
    shard_count = 3,
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_efficientnet",
    size = "large",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=efficientnet"],
    main = "applications_load_weight_test.py",
    shard_count = 8,
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
    deps = [
        ":applications",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "applications_load_weight_test_nasnet",
    srcs = ["applications_load_weight_test.py"],
    args = ["--module=nasnet"],
    main = "applications_load_weight_test.py",
    tags = [
        "no_oss",  # TODO(b/146940090): fix kokoro error
        "no_pip",
    ],
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
