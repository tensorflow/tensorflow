# Description:
#   Contains the Keras API (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "estimator/__init__.py",
        "keras_parameterized.py",
        "ops.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":engine",
        # TODO(scottzhu): Stop exporting the test_utils after removing all
        # callers that using it.
        ":testing_utils",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/keras/applications",
        "//tensorflow/python/keras/datasets",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/mixed_precision/experimental:mixed_precision_experimental",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/premade",
        "//tensorflow/python/keras/preprocessing",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils",
        "//tensorflow/python/keras/wrappers",
        "//tensorflow/python/saved_model",
    ],
)

py_library(
    name = "backend",
    srcs = ["backend.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":backend_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:composite_tensor_utils",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:ctc_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:multi_worker_util",
    ],
)

py_library(
    name = "backend_config",
    srcs = ["backend_config.py"],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow/python:util"],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "engine",
    srcs = [
        ":metrics",
        ":models",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/keras/engine",
    ],
)

py_library(
    name = "activations",
    srcs = [
        "activations.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "base_layer",
    srcs = [],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/keras/engine:base_layer",
    ],
)

py_library(
    name = "callbacks",
    srcs = [
        "callbacks.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/distribute:distributed_file_utils",
        "//tensorflow/python/keras/distribute:multi_worker_training_state",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "combinations",
    srcs = [
        "combinations.py",
    ],
    deps = [
        ":testing_utils",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python:tf2",
    ],
)

py_library(
    name = "callbacks_v1",
    srcs = [
        "callbacks_v1.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/profiler:profiler_v2",
    ],
)

py_library(
    name = "constraints",
    srcs = [
        "constraints.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "initializers",
    srcs = [
        "initializers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "losses",
    srcs = [
        "losses.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "metrics.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":losses",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/losses",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        ":metrics",
        ":optimizers",
        "//tensorflow/python:platform",
        "//tensorflow/python:util",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:version_utils",
    ],
)

py_library(
    name = "optimizers",
    srcs = [
        "optimizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "regularizers",
    srcs = [
        "regularizers.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "testing_utils",
    srcs = [
        "testing_utils.py",
    ],
    deps = [
        ":backend",
        ":models",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf2",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/optimizer_v2",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "activations_test",
    size = "small",
    srcs = ["activations_test.py"],
    python_version = "PY3",
    deps = [
        ":activations",
        ":backend",
        ":combinations",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/layers:advanced_activations",
        "//tensorflow/python/keras/layers:core",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "combinations_test",
    size = "small",
    srcs = ["combinations_test.py"],
    python_version = "PY3",
    deps = [
        ":combinations",
        ":testing_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "constraints_test",
    size = "small",
    srcs = ["constraints_test.py"],
    python_version = "PY3",
    deps = [
        ":backend",
        ":combinations",
        ":constraints",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "initializers_test",
    size = "small",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    deps = [
        ":backend",
        ":combinations",
        ":initializers",
        ":models",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/keras/engine",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "regularizers_test",
    size = "medium",
    srcs = ["regularizers_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "optimizers_test",
    size = "medium",
    srcs = ["optimizers_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "losses_test",
    size = "small",
    srcs = ["losses_test.py"],
    python_version = "PY3",
    deps = [
        ":backend",
        ":combinations",
        ":losses",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/keras/utils:engine_utils",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "metrics_functional_test",
    size = "small",
    srcs = ["metrics_functional_test.py"],
    python_version = "PY3",
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":combinations",
        ":keras",
        ":metrics",
        ":testing_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/training/tracking:util",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "metrics_confusion_matrix_test",
    size = "medium",
    srcs = ["metrics_confusion_matrix_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":combinations",
        ":metrics",
        ":models",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "metrics_correctness_test",
    size = "medium",
    srcs = ["metrics_correctness_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "callbacks_test",
    size = "medium",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_oss",
        "notsan",
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "callbacks_v1_test",
    size = "medium",
    srcs = ["callbacks_v1_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":callbacks",
        ":callbacks_v1",
        ":combinations",
        ":testing_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training_lib",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/utils:np_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "models_test",
    size = "medium",
    srcs = ["models_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_rocm",
        "notsan",  # b/67509773
    ],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "backend_test",
    size = "medium",
    srcs = ["backend_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":backend",
        ":combinations",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:errors",
        "//tensorflow/python:extra_py_tests_deps",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "backend_config_test",
    size = "medium",
    srcs = ["backend_config_test.py"],
    python_version = "PY3",
    deps = [
        ":backend",
        ":backend_config",
        ":combinations",
        "//tensorflow/python:client_testlib",
    ],
)

tf_py_test(
    name = "keras_parameterized_test",
    size = "small",
    srcs = ["keras_parameterized_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    deps = [
        ":keras",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
