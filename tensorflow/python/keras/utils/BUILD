# Description:
#   Contains the Keras Utilities (internal TensorFlow version).

load("//tensorflow:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/feature_column:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow/tools/pip_package:__subpackages__",
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
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":generic_utils",
        ":io_utils",
        ":tf_inspect",
        "//tensorflow/python/framework:tensor",
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
        "//tensorflow/python/framework:tensor_conversion",
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
        "//tensorflow/python/distribute/coordinator:cluster_coordinator",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:nest",
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
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator_py",
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
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/util:nest",
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
        ":engine_utils",
        ":generic_utils",
        ":tf_utils",
        "//tensorflow/python/distribute",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:weights_broadcast_ops",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_util",
        "//tensorflow/python/tpu:tpu_lib",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_library(
    name = "version_utils",
    srcs = [
        "version_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:ops",
    ],
)

py_library(
    name = "np_utils",
    srcs = [
        "np_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
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
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_library(
    name = "tf_inspect",
    srcs = ["tf_inspect.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_library(
    name = "vis_utils",
    srcs = [
        "vis_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "dataset_creator",
    srcs = [
        "dataset_creator.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/types:data",
        "//tensorflow/python/util:tf_export",
    ],
)
