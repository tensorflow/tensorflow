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

# On Windows, python binary is a zip file of runfiles tree.
# Add everything to its data dependency for generating a runfiles tree
# for building the pip package on Windows.
py_binary(
    name = "simple_console_for_windows",
    srcs = ["simple_console_for_windows.py"],
    data = [
        "MANIFEST.in",
        "README",
        "setup.py",
        ":other_headers",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/ndlstm:all_files",
        "//tensorflow/contrib/session_bundle:all_files",
        "//tensorflow/contrib/slim:all_files",
        "//tensorflow/contrib/slim/python/slim/data:all_files",
        "//tensorflow/contrib/slim/python/slim/nets:all_files",
        "//tensorflow/contrib/specs:all_files",
        "//tensorflow/contrib/tensor_forest:all_files",
        "//tensorflow/contrib/tensor_forest/hybrid:all_files",
        "//tensorflow/core:framework_headers",
        "//tensorflow/examples/tutorials/mnist:package",
        "//tensorflow/models/embedding:package",
        "//tensorflow/models/image/alexnet:all_files",
        "//tensorflow/models/image/cifar10:all_files",
        "//tensorflow/models/image/imagenet:all_files",
        "//tensorflow/models/rnn:package",
        "//tensorflow/models/rnn/ptb:package",
        "//tensorflow/models/rnn/translate:package",
        "//tensorflow/python:util_example_parser_configuration",
        "//tensorflow/python/debug:all_files",
        "//tensorflow/python/saved_model:all_files",
        # The following two targets have an issue when archiving them into
        # the python zip, exclude them for now.
        # "//tensorflow/models/image/mnist:convolutional",
        # "//tensorflow/tensorboard",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = select({
        "//tensorflow:windows": [":simple_console_for_windows"],
        "//conditions:default": [
            "MANIFEST.in",
            "README",
            "setup.py",
            ":other_headers",
            ":simple_console",
            "//tensorflow:tensorflow_py",
            "//tensorflow/contrib/ndlstm:all_files",
            "//tensorflow/contrib/session_bundle:all_files",
            "//tensorflow/contrib/slim:all_files",
            "//tensorflow/contrib/slim/python/slim/data:all_files",
            "//tensorflow/contrib/slim/python/slim/nets:all_files",
            "//tensorflow/contrib/specs:all_files",
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
    }),
)
