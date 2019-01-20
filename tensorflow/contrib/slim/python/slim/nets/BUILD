# Description:
#   Contains typical networks definitions.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow_models:__subpackages__",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "nets_pip",
    deps = [
        ":alexnet",
        ":inception",
        ":overfeat",
        ":resnet_v1",
        ":resnet_v2",
        ":vgg",
    ],
)

py_library(
    name = "alexnet",
    srcs = ["alexnet.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "alexnet_test",
    size = "medium",
    srcs = ["alexnet_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":alexnet",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "inception",
    srcs = ["inception.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":inception_v1",
        ":inception_v2",
        ":inception_v3",
    ],
)

py_library(
    name = "inception_v1",
    srcs = ["inception_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "inception_v2",
    srcs = ["inception_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "inception_v3",
    srcs = ["inception_v3.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "inception_v1_test",
    size = "large",
    srcs = ["inception_v1_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception_v1",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:model_analyzer",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "inception_v2_test",
    size = "large",
    srcs = ["inception_v2_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception_v2",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:model_analyzer",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "inception_v3_test",
    size = "large",
    srcs = ["inception_v3_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":inception_v3",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/slim:model_analyzer",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "overfeat",
    srcs = ["overfeat.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "overfeat_test",
    size = "medium",
    srcs = ["overfeat_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":overfeat",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "resnet_utils",
    srcs = ["resnet_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_library(
    name = "resnet_v1",
    srcs = ["resnet_v1.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "resnet_v1_test",
    size = "large",
    srcs = ["resnet_v1_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v1",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "resnet_v2",
    srcs = ["resnet_v2.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "resnet_v2_test",
    size = "large",
    srcs = ["resnet_v2_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":resnet_utils",
        ":resnet_v2",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "vgg",
    srcs = ["vgg.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:variable_scope",
    ],
)

py_test(
    name = "vgg_test",
    size = "medium",
    srcs = ["vgg_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":vgg",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)
