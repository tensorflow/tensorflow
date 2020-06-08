# Description:
# Example TensorFlow models for MNIST used in tutorials

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":input_data",
        ":mnist",
    ],
)

py_library(
    name = "input_data",
    srcs = ["input_data.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "mnist",
    srcs = [
        "mnist.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "fully_connected_feed",
    srcs = [
        "fully_connected_feed.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = ["optonly"],
    deps = [
        ":input_data",
        ":mnist",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "mnist_with_summaries",
    srcs = [
        "mnist_with_summaries.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)

# Note: We need to set the evironment variable to use CPU JIT.
# The way to achieve this is via setting the following:
# TF_XLA_FLAGS='--tf_xla_cpu_global_jit=true'
# before the run command. To use XLA, we also must build
# with --define=with_xla_support=true flag.
# Note (GPU): Add --config=cuda to the build command.
py_binary(
    name = "mnist_softmax_xla",
    srcs = [
        "mnist_softmax_xla.py",
    ],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "fully_connected_feed_test",
    srcs = [
        "fully_connected_feed.py",
    ],
    args = [
        "--fake_data",
        "--max_steps=10",
    ],
    main = "fully_connected_feed.py",
    tags = ["no_pip"],
    deps = [
        ":input_data",
        ":mnist",
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "mnist_with_summaries_test",
    size = "small",
    srcs = [
        "mnist_with_summaries.py",
    ],
    args = [
        "--fake_data",
        "--max_steps=10",
        "--learning_rate=0.00",
    ],
    main = "mnist_with_summaries.py",
    tags = [
        "no_pip",
        "noasan",  # http://b/146080738
        "notsan",  # http://b/29184009
    ],
    deps = [
        ":input_data",
        "//tensorflow:tensorflow_py",
    ],
)
