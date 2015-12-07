# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    deps = ["//tensorflow:tensorflow_py"],
    srcs_version = "PY2AND3",
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "MANIFEST.in",
        "README",
        "setup.py",
        ":simple_console",
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:package",
        "//tensorflow/models/embedding:package",
        "//tensorflow/models/image/cifar10:all_files",
        "//tensorflow/models/image/mnist:convolutional",
        "//tensorflow/models/rnn:package",
        "//tensorflow/models/rnn/ptb:package",
        "//tensorflow/models/rnn/translate:package",
        "//tensorflow/tensorboard",
    ],
)
