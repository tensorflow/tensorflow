load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "xla_sharding",
    srcs = ["xla_sharding.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops:resource_variable_ops",
        "//third_party/py/numpy",
        "@local_xla//xla:xla_data_proto_py",
    ],
)

py_strict_test(
    name = "xla_sharding_test",
    srcs = ["xla_sharding_test.py"],
    deps = [
        ":xla_sharding",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "@local_xla//xla:xla_data_proto_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:absltest",
    ],
)

tpu_py_strict_test(
    name = "resource_variable_xla_sharding_test",
    srcs = ["resource_variable_xla_sharding_test.py"],
    disable_v3_4chips = False,
    tags = ["requires-net:external"],
    deps = [
        ":xla_sharding",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/tpu:device_assignment",
        "//tensorflow/python/tpu:tpu_py",
        "//tensorflow/python/training:adagrad",
    ],
)
