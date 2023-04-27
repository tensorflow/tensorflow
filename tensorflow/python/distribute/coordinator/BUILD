load("//tensorflow:tensorflow.default.bzl", "tf_py_test")
load("//tensorflow/core/platform:distribute.bzl", "distribute_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "cluster_coordinator",
    srcs = ["cluster_coordinator.py"],
    srcs_version = "PY3",
    deps = [
        ":coordinator_context",
        ":metric_utils",
        ":remote_value",
        ":utils",
        ":values",
        ":watchdog",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:func_graph",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:executor",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "@six_archive//:six",
    ],
)

py_library(
    name = "coordinator_context",
    srcs = [
        "coordinator_context.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":remote_value",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "remote_value",
    srcs = ["remote_value.py"],
    srcs_version = "PY3",
    deps = ["//tensorflow/python/util:tf_export"],
)

py_library(
    name = "values",
    srcs = ["values.py"],
    srcs_version = "PY3",
    deps = [
        ":remote_value",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:function",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

distribute_py_test(
    name = "cluster_coordinator_test",
    srcs = ["cluster_coordinator_test.py"],
    python_version = "PY3",
    shard_count = 50,
    tags = [
        "multi_gpu",
        "no_oss",  # TODO(b/214432000): Very flaky under Docker
        "no_pip",
        "noasan",  # TODO(b/171040359): Flaky timeout, even if maximum shards
        "notpu",
        "notsan",  # TODO(b/171040359): Flaky timeout, even if maximum shards
    ],
    xla_tags = [
        "no_cuda_asan",  # Race condition on async test
    ],
    deps = [
        ":cluster_coordinator",
        ":remote_value",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:training_lib",
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "fault_tolerance_test",
    srcs = ["fault_tolerance_test.py"],
    flaky = True,
    python_version = "PY3",
    shard_count = 40,  # = number of tests, so one shard = one test
    tags = [
        "no_oss",  # TODO(b/219580021)
        "noasan",  # Multi-process runner does not work with test sanitizers
        "nomac",  # TODO(b/177065434)
    ],
    deps = [
        ":cluster_coordinator",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:training_lib",
    ],
)

tf_py_test(
    name = "fault_tolerance_coordination_service_test",
    srcs = ["fault_tolerance_coordination_service_test.py"],
    flaky = True,
    python_version = "PY3",
    shard_count = 41,
    tags = [
        # Inherit tags from fault_tolerance_test
        "no_oss",  # TODO(b/219580021)
        "noasan",  # Multi-process runner does not work with test sanitizers
        "nomac",  # TODO(b/177065434)
    ],
    deps = [
        ":fault_tolerance_test",
    ],
)

py_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/eager:monitoring",
        "//tensorflow/python/util:tf_decorator",
    ],
)

tf_py_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":cluster_coordinator",
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
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:training_server_lib",
        "//tensorflow/python/platform:tf_logging",
    ],
)

py_library(
    name = "remote_eager_lib",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)

py_library(
    name = "watchdog",
    srcs = ["watchdog.py"],
    srcs_version = "PY3",
    deps = [],
)

tf_py_test(
    name = "watchdog_test",
    srcs = ["watchdog_test.py"],
    python_version = "PY3",
    tags = [
        "nomac",  # TODO(b/239433962)
    ],
    deps = [
        ":watchdog",
        "//tensorflow/python/eager:test",
    ],
)

tf_py_test(
    name = "get_task_states_test",
    srcs = ["get_task_states_test.py"],
    flaky = True,
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "no_oss",  # TODO(b/219580021)
        "nomac",  # TODO(b/177065434)
    ],
    deps = [
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:parameter_server_strategy_v2",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training:server_lib",
    ],
)
