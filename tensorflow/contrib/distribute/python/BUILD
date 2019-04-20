# Implementation of a prototype TF distributed computation library.

load("//tensorflow/compiler/tests:build_defs.bzl", "tf_xla_py_test")
load("//tensorflow/core:platform/default/distribute.bzl", "distribute_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "distribute_test_lib_pip",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":keras_multi_worker_test_base",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:single_loss_example",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:strategy_test_lib",
        "//tensorflow/python/keras/distribute:keras_correctness_test_lib",
        "//tensorflow/python/keras/distribute:keras_test_lib",
    ],
)

py_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/distribute:mirrored_strategy",
    ],
)

py_library(
    name = "parameter_server_strategy",
    srcs = ["parameter_server_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
    ],
)

cuda_py_test(
    name = "parameter_server_strategy_test",
    srcs = ["parameter_server_strategy_test.py"],
    additional_deps = [
        ":parameter_server_strategy",
        "//tensorflow/python/distribute:central_storage_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_test_lib",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:layers",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

py_library(
    name = "one_device_strategy",
    srcs = ["one_device_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:one_device_strategy",
    ],
)

py_library(
    name = "collective_all_reduce_strategy",
    srcs = ["collective_all_reduce_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
    ],
)

cuda_py_test(
    name = "contrib_mirrored_strategy_test",
    srcs = ["contrib_mirrored_strategy_test.py"],
    additional_deps = [
        ":mirrored_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:values",
    ],
    shard_count = 1,
    tags = [
        "guitar",
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "keras_multi_worker_correctness_test",
    srcs = ["keras_multi_worker_correctness_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        ":mirrored_strategy",
        ":parameter_server_strategy",
        ":keras_multi_worker_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
    ],
    tags = [
        "multi_and_single_gpu",
        "nomsan",  # TODO(b/130299192)
    ],
)

py_library(
    name = "keras_multi_worker_test_base",
    srcs = ["keras_multi_worker_test_base.py"],
    deps = [
        ":collective_all_reduce_strategy",
        ":mirrored_strategy",
        ":parameter_server_strategy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
    ],
)

py_library(
    name = "tpu_strategy",
    srcs = ["tpu_strategy.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/contrib/tpu:tpu_lib",
        "//tensorflow/python/distribute:tpu_strategy",
    ],
)

cuda_py_test(
    name = "collective_all_reduce_strategy_test",
    srcs = ["collective_all_reduce_strategy_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:strategy_test_lib",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:estimator_py",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "optimizer_v2_test",
    srcs = ["optimizer_v2_test.py"],
    additional_deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:single_loss_example",
        ":mirrored_strategy",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/contrib/optimizer_v2:training",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

cuda_py_test(
    name = "estimator_integration_test",
    srcs = ["estimator_integration_test.py"],
    additional_deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/contrib/optimizer_v2:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/feature_column",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # http://b/119349471
        "tf_integration_test",
    ],
)

cuda_py_test(
    name = "estimator_training_test",
    srcs = ["estimator_training_test.py"],
    additional_deps = [
        ":collective_all_reduce_strategy",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        ":mirrored_strategy",
        "//tensorflow/python/distribute:multi_worker_test_base",
        ":parameter_server_strategy",
        "//third_party/py/numpy",
        "//tensorflow/contrib/optimizer_v2:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_config",
        "//tensorflow/python/distribute:distribute_coordinator",
        "//tensorflow/python/distribute:distribute_coordinator_context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/feature_column",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
    ],
    shard_count = 48,
    tags = [
        "multi_and_single_gpu",
        # TODO(b/118768923): Re-enable {a,m,t}san test.
        "noasan",
        "nomsan",
        "notsan",
        "no_oss",  # http://b/119349471
    ],
)

py_library(
    name = "monitor",
    srcs = ["monitor.py"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "monitor_test",
    srcs = ["monitor_test.py"],
    additional_deps = [
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:strategy_combinations",
        ":monitor",
        "//tensorflow/python/distribute:single_loss_example",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python/distribute:one_device_strategy",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
    ],
    tags = [
        "multi_and_single_gpu",
    ],
)

# TODO(b/121200287): Remove this in 2.0
distribute_py_test(
    name = "keras_backward_compat_test",
    srcs = ["keras_backward_compat_test.py"],
    full_precision = True,
    main = "keras_backward_compat_test.py",
    shard_count = 31,
    tags = [
        "multi_and_single_gpu",
        "no_windows_gpu",
        "notsan",
    ],
    deps = [
        ":mirrored_strategy",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/keras/distribute:keras_test_lib",
    ],
)
