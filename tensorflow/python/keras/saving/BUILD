# Description:
#   Contains the Keras save model API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

filegroup(
    name = "all_py_srcs",
    srcs = glob([
        "*.py",
        "**/*.py",
    ]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "saving",
    srcs = [
        "__init__.py",
        "hdf5_format.py",
        "model_config.py",
        "save.py",
        "saved_model/base_serialization.py",
        "saved_model/constants.py",
        "saved_model/json_utils.py",
        "saved_model/layer_serialization.py",
        "saved_model/load.py",
        "saved_model/metric_serialization.py",
        "saved_model/model_serialization.py",
        "saved_model/network_serialization.py",
        "saved_model/save.py",
        "saved_model/save_impl.py",
        "saved_model/serialized_attributes.py",
        "saved_model/utils.py",
        "saved_model_experimental.py",
        "saving_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":load_context",
        "//tensorflow/python:lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:saver",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:optimizers",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/mixed_precision:autocast_variable",
        "//tensorflow/python/keras/protobuf:saved_metadata_proto_py",
        "//tensorflow/python/keras/saving/utils_v1",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/training/tracking",
    ],
)

py_library(
    name = "load_context",
    srcs = [
        "saved_model/load_context.py",
    ],
    srcs_version = "PY3",
    deps = [],
)

tf_py_test(
    name = "metrics_serialization_test",
    size = "medium",
    srcs = ["metrics_serialization_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "notsan",  # TODO(b/170870790)
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "losses_serialization_test",
    size = "medium",
    srcs = ["losses_serialization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "save_weights_test",
    size = "medium",
    srcs = ["save_weights_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss_py35",  # b/147011479
        "no_windows",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "save_test",
    size = "medium",
    srcs = ["save_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/feature_column:feature_column_v2",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "saved_model_experimental_test",
    size = "medium",
    srcs = ["saved_model_experimental_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",  # TODO(b/119349471): Re-enable
        "no_windows",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "saved_model_test",
    size = "medium",
    srcs = ["saved_model/saved_model_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_rocm",
        "no_windows",
        "notsan",  #TODO(b/181771982): it is flaky
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "saving_utils_test",
    size = "medium",
    srcs = ["saving_utils_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "revive_test",
    size = "medium",
    srcs = ["saved_model/revive_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_windows",  # b/158005583
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "json_utils_test",
    size = "small",
    srcs = ["saved_model/json_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":saving",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
