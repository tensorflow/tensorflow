# Description:
#   Contains the Keras normalization layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/feature_column:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/training/tracking:__pkg__",
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow_models/official/vision/beta/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "normalization",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":batch_normalization",
        ":batch_normalization_v1",
        ":layer_normalization",
    ],
)

py_library(
    name = "batch_normalization",
    srcs = ["batch_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:control_flow_util",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "batch_normalization_v1",
    srcs = ["batch_normalization_v1.py"],
    srcs_version = "PY3",
    deps = [
        ":batch_normalization",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "layer_normalization",
    srcs = ["layer_normalization.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/util:tf_export",
    ],
)

cuda_py_test(
    name = "batch_normalization_test",
    size = "medium",
    srcs = ["batch_normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        ":batch_normalization",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:wrap_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/layers",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "layer_normalization_test",
    size = "medium",
    srcs = ["layer_normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "notsan",
    ],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
