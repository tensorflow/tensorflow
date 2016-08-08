# Description:
# TensorFlow is a computational framework, primarily for use in machine
# learning applications.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files([
    "LICENSE",
    "ACKNOWLEDGMENTS",
])

# Config setting for determining if we are building for Android.
config_setting(
    name = "android",
    values = {
        "crosstool_top": "//external:android/crosstool",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "android_arm",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "android_cpu": "armeabi-v7a",
    },
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    values = {"cpu": "darwin"},
    visibility = ["//visibility:public"],
)

package_group(
    name = "internal",
    packages = [
        "//learning/vis/...",
        "//tensorflow/...",
    ],
)

sh_binary(
    name = "swig",
    srcs = ["tools/swig/swig.sh"],
    data = glob(["tools/swig/**"]),
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "g3doc/sitemap.md",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "tensorflow_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/python"],
)

filegroup(
    name = "all_opensource_files",
    data = [
        ":all_files",
        "//tensorflow/c:all_files",
        "//tensorflow/cc:all_files",
        "//tensorflow/contrib:all_files",
        "//tensorflow/contrib/copy_graph:all_files",
        "//tensorflow/contrib/distributions:all_files",
        "//tensorflow/contrib/factorization:all_files",
        "//tensorflow/contrib/factorization/kernels:all_files",
        "//tensorflow/contrib/ffmpeg:all_files",
        "//tensorflow/contrib/ffmpeg/default:all_files",
        "//tensorflow/contrib/framework:all_files",
        "//tensorflow/contrib/graph_editor:all_files",
        "//tensorflow/contrib/layers:all_files",
        "//tensorflow/contrib/layers/kernels:all_files",
        "//tensorflow/contrib/learn:all_files",
        "//tensorflow/contrib/learn/python/learn/datasets:all_files",
        "//tensorflow/contrib/linear_optimizer:all_files",
        "//tensorflow/contrib/linear_optimizer/kernels:all_files",
        "//tensorflow/contrib/lookup:all_files",
        "//tensorflow/contrib/losses:all_files",
        "//tensorflow/contrib/metrics:all_files",
        "//tensorflow/contrib/metrics/kernels:all_files",
        "//tensorflow/contrib/opt:all_files",
        "//tensorflow/contrib/quantization:all_files",
        "//tensorflow/contrib/quantization/kernels:all_files",
        "//tensorflow/contrib/quantization/tools:all_files",
        "//tensorflow/contrib/session_bundle:all_files",
        "//tensorflow/contrib/session_bundle/example:all_files",
        "//tensorflow/contrib/slim:all_files",
        "//tensorflow/contrib/slim/python/slim/data:all_files",
        "//tensorflow/contrib/tensor_forest:all_files",
        "//tensorflow/contrib/testing:all_files",
        "//tensorflow/contrib/util:all_files",
        "//tensorflow/core:all_files",
        "//tensorflow/core/debug:all_files",
        "//tensorflow/core/distributed_runtime:all_files",
        "//tensorflow/core/distributed_runtime/rpc:all_files",
        "//tensorflow/core/kernels:all_files",
        "//tensorflow/core/ops/compat:all_files",
        "//tensorflow/core/platform/cloud:all_files",
        "//tensorflow/core/platform/default/build_config:all_files",
        "//tensorflow/core/util/ctc:all_files",
        "//tensorflow/examples/android:all_files",
        "//tensorflow/examples/how_tos/reading_data:all_files",
        "//tensorflow/examples/image_retraining:all_files",
        "//tensorflow/examples/label_image:all_files",
        "//tensorflow/examples/learn:all_files",
        "//tensorflow/examples/skflow:all_files",
        "//tensorflow/examples/tutorials/mnist:all_files",
        "//tensorflow/examples/tutorials/word2vec:all_files",
        "//tensorflow/g3doc/how_tos/adding_an_op:all_files",
        "//tensorflow/g3doc/tutorials:all_files",
        "//tensorflow/models/embedding:all_files",
        "//tensorflow/models/image/alexnet:all_files",
        "//tensorflow/models/image/cifar10:all_files",
        "//tensorflow/models/image/imagenet:all_files",
        "//tensorflow/models/image/mnist:all_files",
        "//tensorflow/models/rnn:all_files",
        "//tensorflow/models/rnn/ptb:all_files",
        "//tensorflow/models/rnn/translate:all_files",
        "//tensorflow/python:all_files",
        "//tensorflow/python/kernel_tests:all_files",
        "//tensorflow/python/tools:all_files",
        "//tensorflow/tensorboard:all_files",
        "//tensorflow/tensorboard/app:all_files",
        "//tensorflow/tensorboard/backend:all_files",
        "//tensorflow/tensorboard/components:all_files",
        "//tensorflow/tensorboard/lib:all_files",
        "//tensorflow/tensorboard/lib/python:all_files",
        "//tensorflow/tensorboard/scripts:all_files",
        "//tensorflow/tools/dist_test/server:all_files",
        "//tensorflow/tools/docker:all_files",
        "//tensorflow/tools/docker/notebooks:all_files",
        "//tensorflow/tools/docs:all_files",
        "//tensorflow/tools/proto_text:all_files",
        "//tensorflow/tools/test:all_files",
        "//tensorflow/user_ops:all_files",
    ],
    visibility = [":__subpackages__"],
)

# -------------------------------------------
# New rules should be added above this target.
# -------------------------------------------
cc_binary(
    name = "libtensorflow.so",
    linkshared = 1,
    deps = [
        "//tensorflow/core:tensorflow",
    ],
)

cc_binary(
    name = "libtensorflow_cc.so",
    linkshared = 1,
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:tensorflow",
    ],
)
