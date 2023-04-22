# Description:
#   Contains the Keras Utilities (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/feature_column:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "utils",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":all_utils",
    ],
)

py_library(
    name = "all_utils",
    srcs = [
        "all_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":control_flow_util",
        ":engine_utils",
        ":generic_utils",
        ":layer_utils",
        ":multi_gpu_utils",
        ":np_utils",
        ":vis_utils",
    ],
)

py_library(
    name = "control_flow_util",
    srcs = ["control_flow_util.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "kpl_test_utils",
    srcs = ["kpl_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/layers/preprocessing:string_lookup",
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":generic_utils",
        ":io_utils",
        ":tf_inspect",
    ],
)

py_library(
    name = "engine_utils",
    srcs = [
        "conv_utils.py",
        "losses_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":data_utils",
        ":io_utils",
        "//tensorflow/python/keras:backend",
    ],
)

py_library(
    name = "io_utils",
    srcs = ["io_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tf_utils",
    srcs = ["tf_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":object_identity",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:smart_cond",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "generic_utils",
    srcs = [
        "generic_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":tf_contextlib",
        ":tf_inspect",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "mode_keys",
    srcs = [
        "mode_keys.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/saved_model/model_utils:mode_keys",
    ],
)

py_library(
    name = "layer_utils",
    srcs = [
        "kernelized_utils.py",
        "layer_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":engine_utils",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "metrics_utils",
    srcs = [
        "metrics_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":generic_utils",
        ":tf_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:distribute",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_util",
        "//tensorflow/python/tpu:tpu_lib",
    ],
)

py_library(
    name = "version_utils",
    srcs = [
        "version_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "multi_gpu_utils",
    srcs = [
        "multi_gpu_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/layers",
    ],
)

py_library(
    name = "np_utils",
    srcs = [
        "np_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "object_identity",
    srcs = ["object_identity.py"],
    srcs_version = "PY3",
    deps = [],
)

py_library(
    name = "tf_contextlib",
    srcs = ["tf_contextlib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "vis_utils",
    srcs = [
        "vis_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "dataset_creator",
    srcs = [
        "dataset_creator.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_test(
    name = "dataset_creator_test",
    srcs = ["dataset_creator_test.py"],
    python_version = "PY3",
    deps = [
        ":dataset_creator",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/keras/distribute:multi_worker_testing_utils",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/layers:core",
    ],
)

tf_py_test(
    name = "data_utils_test",
    size = "medium",
    srcs = ["data_utils_test.py"],
    python_version = "PY3",
    shard_count = 6,
    tags = [
        "noasan",  # times out
        "notsan",
        "optonly",  # times out
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "generic_utils_test",
    size = "small",
    srcs = ["generic_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":generic_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "version_utils_test",
    size = "small",
    srcs = ["version_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":version_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "tf_utils_test",
    size = "small",
    srcs = ["tf_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":tf_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
    ],
)

tf_py_test(
    name = "composite_tensor_support_test",
    size = "medium",
    srcs = ["composite_tensor_support_test.py"],
    python_version = "PY3",
    shard_count = 8,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "io_utils_test",
    size = "small",
    srcs = ["io_utils_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  # TODO: needs investigation on Windows
        "notsan",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "layer_utils_test",
    size = "small",
    srcs = ["layer_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":layer_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/training/tracking",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "np_utils_test",
    size = "small",
    srcs = ["np_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "kernelized_utils_test",
    size = "small",
    srcs = ["kernelized_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":layer_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:layers",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "multi_gpu_utils_test",
    srcs = ["multi_gpu_utils_test.py"],
    python_version = "PY3",
    tags = [
        "guitar",
        "multi_gpu",
    ],
    xla_enable_strict_auto_jit = True,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "vis_utils_test",
    size = "small",
    srcs = ["vis_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "conv_utils_test",
    size = "small",
    srcs = ["conv_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "metrics_utils_test",
    size = "small",
    srcs = ["metrics_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/user_ops:ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "losses_utils_test",
    size = "small",
    srcs = ["losses_utils_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//tensorflow/python/ops/ragged:ragged_array_ops",
        "//tensorflow/python/ops/ragged:ragged_concat_ops",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/user_ops:ops",
    ],
)
