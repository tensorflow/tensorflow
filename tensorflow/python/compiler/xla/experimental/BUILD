load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_strict_library(
    name = "xla_sharding",
    srcs = ["xla_sharding.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla/python:xla",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/core:protos_all_py",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "xla_sharding_test",
    srcs = ["xla_sharding_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":xla_sharding",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
    ],
)
