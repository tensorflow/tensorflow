# Python bindings for RPC client and server ops.

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "rpc_ops",
    srcs = [
        "rpc_ops.py",
    ],
    deps = [
        "//tensorflow/distribute/experimental/rpc/kernels:gen_rpc_ops",
        "//tensorflow/distribute/experimental/rpc/proto:tf_rpc_service_proto_py",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:none_tensor",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "rpc_ops_test",
    size = "medium",
    srcs = ["rpc_ops_test.py"],
    shard_count = 7,
    tags = [
        "no_mac",  # flaky, b/205156709
    ],
    deps = [
        ":rpc_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "@pypi_portpicker//:pkg",
    ],
)
