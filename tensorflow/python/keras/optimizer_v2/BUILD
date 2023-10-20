# Description:
#   Contains the Keras OptimizerV2 API (internal TensorFlow version).

load("//tensorflow:py.default.bzl", "py_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python:__pkg__",
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/tpu/tests:__pkg__",
        "//tensorflow/python/trackable:__pkg__",
        "//tensorflow/python/training:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "optimizer_v2",
    srcs = [
        "adadelta.py",
        "adagrad.py",
        "adam.py",
        "adamax.py",
        "ftrl.py",
        "gradient_descent.py",
        "nadam.py",
        "optimizer_v2.py",
        "rmsprop.py",
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":learning_rate_schedule",
        "//tensorflow/python/distribute:central_storage_strategy",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:backend_config",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variables",
    ],
)

py_library(
    name = "learning_rate_schedule",
    srcs = [
        "learning_rate_schedule.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_case",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
    ],
)

py_library(
    name = "legacy_learning_rate_decay",
    srcs = ["legacy_learning_rate_decay.py"],
    srcs_version = "PY3",
    deps = [
        ":learning_rate_schedule",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)
