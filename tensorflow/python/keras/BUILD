# Description:
#   Contains the Keras API (internal TensorFlow version).

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = [
        "//tensorflow/python/keras/google/private_tf_api_test:__pkg__",
    ],
)

config_setting(
    name = "no_keras_py_deps",
    define_values = {"no_keras_py_deps": "true"},
    visibility = ["//visibility:public"],
)

py_library(
    name = "keras",
    srcs = [
        "__init__.py",
        "keras_parameterized.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":engine",
        ":testing_utils",  # TODO(scottzhu): Stop exporting the test_utils after removing all the callers.
        "//tensorflow/python:training",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/mixed_precision:mixed_precision_experimental",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "backend",
    srcs = ["backend.py"],
    srcs_version = "PY3",
    deps = [
        ":backend_config",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
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
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras/distribute:distribute_coordinator_utils",
        "//tensorflow/python/keras/engine:keras_tensor",
        "//tensorflow/python/keras/utils:control_flow_util",
        "//tensorflow/python/keras/utils:object_identity",
        "//tensorflow/python/keras/utils:tf_contextlib",
        "//tensorflow/python/keras/utils:tf_inspect",
        "//tensorflow/python/summary",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "backend_config",
    srcs = ["backend_config.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "engine",
    srcs = [
        ":metrics",
        ":models",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/keras/engine",
    ],
)

py_library(
    name = "activations",
    srcs = [
        "activations.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/layers:advanced_activations",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

# TODO(scottzhu): Cleanup this target and point all the user to keras/engine.
py_library(
    name = "base_layer",
    srcs = [],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/keras/engine:base_layer",
    ],
)

py_library(
    name = "callbacks",
    srcs = [
        "callbacks.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python:errors",
        "//tensorflow/python/keras/distribute:distributed_file_utils",
        "//tensorflow/python/keras/distribute:worker_training_state",
        "//tensorflow/python/keras/protobuf:projector_config_proto_py",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:mode_keys",
        "//tensorflow/python/profiler:profiler_v2",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
    ],
)

py_library(
    name = "combinations",
    srcs = [
        "combinations.py",
    ],
    srcs_version = "PY3",
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
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python:errors",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/profiler:profiler_v2",
    ],
)

py_library(
    name = "constraints",
    srcs = [
        "constraints.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

py_library(
    name = "losses",
    srcs = [
        "losses.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python:cond",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/util:dispatch",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "metrics.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":activations",
        ":backend",
        ":losses",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:confusion_matrix",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variables",
        "//tensorflow/python:weights_broadcast_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras/distribute",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:metrics_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/losses",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "models",
    srcs = [
        "models.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":metrics",
        ":optimizers",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:version_utils",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "optimizers",
    srcs = [
        "optimizer_v1.py",
        "optimizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "regularizers",
    srcs = [
        "regularizers.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        "//tensorflow/python/keras/utils:engine_utils",
    ],
)

# Internally urllib.request.urlretrieve library requires Google
# SSL context to be provided to work in python 3. This isn't needed in OSS.
# copybara:uncomment_begin(google-only)
# py_library(
#     name = "url_utils",
#     srcs = ["google/url_utils.py"],
#     srcs_version = "PY3",
#     deps = ["//pyglib/contrib/google_ssl"],
# )
# copybara:uncomment_end

py_library(
    name = "testing_utils",
    srcs = [
        "testing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":backend",
        ":models",
        "//tensorflow/python:config",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/layers",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
    ],
)
