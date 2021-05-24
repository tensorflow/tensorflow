# Description:
#   Contains the Keras save model API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
    ],
    licenses = ["notice"],
)

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
        "saved_model_experimental.py",
        "saving_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
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
        "//tensorflow/python/keras/saving/saved_model",
        "//tensorflow/python/keras/saving/utils_v1",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/training/tracking",
    ],
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
