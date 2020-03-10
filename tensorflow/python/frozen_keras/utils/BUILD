load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "tf_utils",
    srcs = ["tf_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:smart_cond",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/frozen_keras:backend",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "conv_utils",
    srcs = [
        "conv_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/frozen_keras:backend",
    ],
)

py_library(
    name = "generic_utils",
    srcs = [
        "generic_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "layer_utils",
    srcs = [
        "layer_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":conv_utils",
        "//tensorflow/python:util",
        "//tensorflow/python/frozen_keras:backend",
        "//third_party/py/numpy",
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
        "//tensorflow/python/frozen_keras:regularizers",
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
    ],
)

tf_py_test(
    name = "conv_utils_test",
    size = "small",
    srcs = ["conv_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":conv_utils",
        "//tensorflow/python:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
