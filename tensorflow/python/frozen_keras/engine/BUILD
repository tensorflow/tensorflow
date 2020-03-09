load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:__subpackages__"],
    licenses = ["notice"],  # Apache 2.0
)

#TODO(scottzhu): Cleanup all the deps to python/keras

py_library(
    name = "legacy_base_layer",
    srcs = ["legacy_base_layer.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:auto_control_deps",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/autograph/core",
        "//tensorflow/python/autograph/impl",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/frozen_keras:constraint",
        "//tensorflow/python/frozen_keras:initializers",
        "//tensorflow/python/frozen_keras:regularizers",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras:metrics",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/engine:base_layer_utils",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/saving",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/module",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/training/tracking",
        "//tensorflow/python/training/tracking:base",
        "//tensorflow/python/training/tracking:data_structures",
        "//tensorflow/python/training/tracking:layer_utils",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "legacy_base_layer_test",
    size = "medium",
    srcs = ["legacy_base_layer_test.py"],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_rocm",
        "nomac",  # TODO(mihaimaruseac): b/127695564
    ],
    deps = [
        ":legacy_base_layer",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
