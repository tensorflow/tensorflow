# tf.data service tests.

load("//tensorflow:tensorflow.default.bzl", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_py_test(
    name = "auto_shard_test",
    size = "medium",
    srcs = ["auto_shard_test.py"],
    shard_count = 32,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "coordinated_read_ft_test",
    size = "medium",
    srcs = ["coordinated_read_ft_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "coordinated_read_test",
    size = "medium",
    srcs = ["coordinated_read_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "data_service_ops_test",
    size = "medium",
    srcs = ["data_service_ops_test.py"],
    shard_count = 32,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "dynamic_sharding_test",
    size = "medium",
    srcs = ["dynamic_sharding_test.py"],
    shard_count = 16,
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

tf_py_test(
    name = "fault_tolerance_test",
    size = "small",
    srcs = ["fault_tolerance_test.py"],
    shard_count = 8,
    tags = [],
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "local_workers_test",
    size = "medium",
    srcs = ["local_workers_test.py"],
    shard_count = 24,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "metadata_test",
    size = "medium",
    srcs = ["metadata_test.py"],
    shard_count = 2,
    deps = [
        ":test_base",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "cross_trainer_cache_test",
    size = "medium",
    srcs = ["cross_trainer_cache_test.py"],
    tags = [],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "cross_trainer_cache_ft_test",
    size = "medium",
    srcs = ["cross_trainer_cache_ft_test.py"],
    shard_count = 4,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "distributed_save_test",
    size = "medium",
    srcs = ["distributed_save_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python/data/experimental/kernel_tests/service:test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "distributed_save_ft_test",
    size = "medium",
    srcs = ["distributed_save_ft_test.py"],
    shard_count = 17,
    deps = [
        ":test_base",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
    ],
)

tf_py_test(
    name = "multi_device_test",
    size = "small",
    srcs = ["multi_device_test.py"],
    deps = [
        ":test_base",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

py_library(
    name = "multi_process_cluster",
    srcs = ["multi_process_cluster.py"],
    srcs_version = "PY3",
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/distribute:multi_process_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_test(
    name = "multi_process_cluster_test",
    size = "medium",
    srcs = ["multi_process_cluster_test.py"],
    shard_count = 3,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "worker_tags_test",
    size = "medium",
    srcs = ["worker_tags_test.py"],
    shard_count = 16,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)
