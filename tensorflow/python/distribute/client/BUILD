load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "client",
    srcs = ["client.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":metric_utils",
        ":utils",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:executor",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:remote",
        "@absl_py//absl/logging",
        "@six_archive//:six",
    ],
)

tf_py_test(
    name = "client_test",
    size = "small",
    srcs = ["client_test.py"],
    python_version = "PY3",
    shard_count = 12,
    deps = [
        ":client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/logging",
    ],
)

tf_py_test(
    name = "parameter_server_client_test",
    srcs = ["parameter_server_client_test.py"],
    python_version = "PY3",
    shard_count = 14,
    tags = ["no_oss"],  # TODO(b/162119374)
    deps = [
        ":client",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/logging",
    ],
)

tf_py_test(
    name = "client_mpr_test",
    srcs = ["client_mpr_test.py"],
    python_version = "PY3",
    shard_count = 2,
    tags = ["no_oss"],  # TODO(b/162119374)
    deps = [
        ":remote_eager_lib",
        ":utils",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:sharded_variable",
        "//tensorflow/python/distribute/client",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python/eager:monitoring",
    ],
)

tf_py_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":client",
        ":metric_utils",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:test",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:training_server_lib",
    ],
)

py_library(
    name = "remote_eager_lib",
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
