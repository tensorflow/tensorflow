# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

load("//tensorflow:tensorflow.bzl", "transitive_hdrs")

transitive_hdrs(
    name = "other_headers",
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "README",
        "setup.py",
        ":other_headers",
        ":simple_console",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/session_bundle:all_files",
        "//tensorflow/contrib/slim:all_files",
        "//tensorflow/contrib/slim/python/slim/data:all_files",
        "//tensorflow/contrib/slim/python/slim/nets:all_files",
        "//tensorflow/contrib/tensor_forest:all_files",
        "//tensorflow/contrib/tensor_forest/hybrid:all_files",
        "//tensorflow/core:framework_headers",
        "//tensorflow/examples/tutorials/mnist:package",
        "//tensorflow/models/embedding:package",
        "//tensorflow/models/image/alexnet:all_files",
        "//tensorflow/models/image/cifar10:all_files",
        "//tensorflow/models/image/imagenet:all_files",
        "//tensorflow/models/image/mnist:convolutional",
        "//tensorflow/models/rnn:package",
        "//tensorflow/models/rnn/ptb:package",
        "//tensorflow/models/rnn/translate:package",
        "//tensorflow/python:util_example_parser_configuration",
        "//tensorflow/python/debug:all_files",
        "//tensorflow/python/saved_model:all_files",
        "//tensorflow/tensorboard",
    ],
)
