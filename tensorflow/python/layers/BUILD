# python/layers package

load("//tensorflow:strict.default.bzl", "py_strict_library")

visibility = [
    "//tensorflow:__subpackages__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = visibility,
    licenses = ["notice"],
)

py_strict_library(
    name = "layers_base",
    srcs = [
        "__init__.py",
        "base.py",
    ],
    srcs_version = "PY3",
    visibility = visibility + [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
        "//third_party/mlperf:__subpackages__",
    ],
    deps = [
        "//tensorflow/python/keras/legacy_tf_layers:layers_base",
    ],
)

py_strict_library(
    name = "layers_util",
    srcs = [
        "utils.py",
    ],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/framework:smart_cond",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:variables",
    ],
)

py_strict_library(
    name = "layers",
    srcs = [
        "convolutional.py",
        "core.py",
        "layers.py",
        "normalization.py",
        "pooling.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//third_party/cloud_tpu/convergence_tools:__subpackages__",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":layers_base",
        "//tensorflow/python/keras/legacy_tf_layers:convolutional",
        "//tensorflow/python/keras/legacy_tf_layers:core",
        "//tensorflow/python/keras/legacy_tf_layers:pooling",
        "//tensorflow/python/util:lazy_loader",
        # Normalization layer will need //third_party/py/tf_keras/legacy_tf_layers:normalization
        # Client lib should import that, since this target can't import it due to
        # circular dependency.
    ],
)
