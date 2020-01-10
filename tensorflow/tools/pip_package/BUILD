# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

py_binary(
    name = "simple_console",
    srcs = ["simple_console.py"],
    deps = ["//tensorflow:tensorflow_py"],
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
        "//tensorflow/models/image/cifar10:cifar10_eval",
        "//tensorflow/models/image/cifar10:cifar10_train",
        "//tensorflow/models/image/cifar10:cifar10_multi_gpu_train",
        "//tensorflow/models/image/mnist:convolutional",
        "//tensorflow/models/rnn:package",
        "//tensorflow/tensorboard",
    ],
)
