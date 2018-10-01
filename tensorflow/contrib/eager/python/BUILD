licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "tfe",
    srcs = ["tfe.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":datasets",
        ":metrics",
        ":network",
        ":parameter_server",
        ":remote",
        ":saver",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:execution_callbacks",
        "//tensorflow/python/eager:function",
    ],
)

cuda_py_test(
    name = "tfe_test",
    srcs = ["tfe_test.py"],
    additional_deps = [
        ":tfe",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:metrics",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:summary",
    ],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/contrib/data/python/ops:prefetching_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    additional_deps = [
        ":datasets",
        "//tensorflow/contrib/data/python/ops:prefetching_ops",
        "//tensorflow/contrib/data/python/ops:threadpool",
        "//tensorflow/contrib/data/python/ops:unique",
        "//tensorflow/contrib/lookup:lookup_py",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data",
        "//tensorflow/python/eager:test",
    ],
    tags = ["noguitar"],
)

py_library(
    name = "saver",
    srcs = ["saver.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "parameter_server",
    srcs = ["parameter_server.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "saver_test",
    srcs = ["saver_test.py"],
    additional_deps = [
        ":saver",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "metrics",
    srcs = [
        "metrics.py",
        "metrics_impl.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/training/checkpointable:base",
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metrics",
        "//tensorflow/contrib/summary:summary_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python:training",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "evaluator",
    srcs = [
        "evaluator.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":datasets",
        ":metrics",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:function",
        "@six_archive//:six",
    ],
)

py_test(
    name = "evaluator_test",
    srcs = ["evaluator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":evaluator",
        ":metrics",
        "//tensorflow/contrib/summary:summary_test_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "network",
    srcs = ["network.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
    ],
)

py_test(
    name = "network_test",
    srcs = ["network_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":network",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "remote",
    srcs = ["remote.py"],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "remote_test",
    srcs = ["remote_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":parameter_server",
        ":remote",
        "//tensorflow/contrib/eager/python:tfe",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:function",
    ],
)
