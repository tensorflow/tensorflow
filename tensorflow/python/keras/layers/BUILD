# Description:
#   Contains the Keras layers (internal TensorFlow version).

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # TODO(scottzhu): Remove non-keras deps from TF.
    default_visibility = [
        "//tensorflow/python/distribute:__pkg__",
        "//tensorflow/python/feature_column:__pkg__",
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/python/training/tracking:__pkg__",
        "//tensorflow/tools/pip_package:__pkg__",
        "//tensorflow_models/official/projects/residual_mobilenet/modeling/backbones:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

# A separate build for layers without serialization to avoid circular deps
# with feature column.
py_library(
    name = "layers",
    srcs = [
        "__init__.py",
        "serialization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":advanced_activations",
        ":convolutional",
        ":convolutional_recurrent",
        ":core",
        ":dense_attention",
        ":embeddings",
        ":merge",
        ":pooling",
        ":recurrent",
        ":rnn_cell_wrapper_v2",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/utils:tf_utils",
    ],
)

py_library(
    name = "advanced_activations",
    srcs = ["advanced_activations.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:math_ops",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "convolutional",
    srcs = ["convolutional.py"],
    srcs_version = "PY3",
    deps = [
        ":pooling",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "convolutional_recurrent",
    srcs = ["convolutional_recurrent.py"],
    srcs_version = "PY3",
    deps = [
        ":recurrent",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "core",
    srcs = ["core.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:standard_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:losses",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/engine:keras_tensor",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dense_attention",
    srcs = ["dense_attention.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "embeddings",
    srcs = ["embeddings.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:base_layer",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "merge",
    srcs = ["merge.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "pooling",
    srcs = ["pooling.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:engine_utils",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "recurrent",
    srcs = ["recurrent.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:cond",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:activations",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:base_layer",
        "//tensorflow/python/keras:constraints",
        "//tensorflow/python/keras:regularizers",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/initializers",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/trackable:base",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "rnn_cell_wrapper_v2",
    srcs = ["rnn_cell_wrapper_v2.py"],
    srcs_version = "PY3",
    deps = [
        ":recurrent",
        "//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_wrapper_impl",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
)

