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
    python_version = "PY3",
    srcs_version = "PY3",
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
