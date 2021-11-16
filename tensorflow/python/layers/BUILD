# python/layers package

visibility = [
    "//tensorflow:__subpackages__",
]

package(
    default_visibility = visibility,
    licenses = ["notice"],
)

py_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/keras/legacy_tf_layers:layers_base",
    ],
)

py_library(
    name = "layers_util",
    srcs = [
        "utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:smart_cond",
    ],
)

py_library(
    name = "layers",
    srcs = [
        "convolutional.py",
        "core.py",
        "layers.py",
        "normalization.py",
        "pooling.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":layers_base",
        "//tensorflow/python/keras/legacy_tf_layers:convolutional",
        "//tensorflow/python/keras/legacy_tf_layers:core",
        "//tensorflow/python/keras/legacy_tf_layers:pooling",
        # Normalization layer will need //third_party/py/keras/legacy_tf_layers:normalization
        # Client lib should import that, since this target can't import it due to
        # circular dependency.
    ],
)
