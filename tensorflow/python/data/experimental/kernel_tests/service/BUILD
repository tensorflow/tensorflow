# tf.data service tests.

load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "auto_shard_test",
    size = "medium",
    srcs = ["auto_shard_test.py"],
    shard_count = 32,
    tags = [
        "no_oss",  # TODO(b/289230931): Investigate flaky test.
    ],
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "coordinated_read_ft_test",
    size = "medium",
    srcs = ["coordinated_read_ft_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "coordinated_read_test",
    size = "medium",
    srcs = ["coordinated_read_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "data_service_ops_test",
    size = "medium",
    srcs = ["data_service_ops_test.py"],
    shard_count = 32,
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:variable_scope",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "gpu_test",
    srcs = ["gpu_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:experimental_dataset_ops_gen",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "dynamic_sharding_test",
    size = "medium",
    srcs = ["dynamic_sharding_test.py"],
    shard_count = 16,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "fault_tolerance_test",
    size = "small",
    srcs = ["fault_tolerance_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
        "@pypi_portpicker//:pkg",
    ],
)

tf_py_strict_test(
    name = "local_workers_test",
    size = "medium",
    srcs = ["local_workers_test.py"],
    shard_count = 24,
    tags = [
        "no_oss",  # TODO(b/295501569)
    ],
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

tf_py_strict_test(
    name = "metadata_test",
    size = "medium",
    srcs = ["metadata_test.py"],
    shard_count = 2,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:script_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "cross_trainer_cache_test",
    size = "medium",
    srcs = ["cross_trainer_cache_test.py"],
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
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
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "distributed_save_test",
    size = "medium",
    srcs = ["distributed_save_test.py"],
    # copybara:uncomment_begin(google-only)
    # exec_properties = select({
    # "//tools/cpp:msan_build": {"mem": "20g"},
    # "//conditions:default": None,
    # }),
    # copybara:uncomment_end
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/experimental/service:_pywrap_snapshot_utils",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "distributed_save_load_test",
    size = "medium",
    srcs = ["distributed_save_load_test.py"],
    shard_count = 8,
    tags = [
        "requires-mem:20g",
    ],
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "distributed_save_load_ft_test",
    size = "medium",
    srcs = ["distributed_save_load_ft_test.py"],
    # copybara:uncomment_begin(google-only)
    # exec_properties = select({
    # "//tools/cpp:msan_build": {"mem": "24g"},
    # "//conditions:default": {"mem": "20g"},
    # }),
    # copybara:uncomment_end
    shard_count = 8,
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "distributed_save_ft_test",
    size = "medium",
    srcs = ["distributed_save_ft_test.py"],
    # copybara:uncomment_begin(google-only)
    # exec_properties = select({
    # "//tools/cpp:msan_build": {"mem": "20g"},
    # "//conditions:default": None,
    # }),
    # copybara:uncomment_end
    shard_count = 17,
    tags = [
        "no_mac",  # TODO(b/290355883): Fix the flakyness in macos
    ],
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:distributed_save_op",
        "//tensorflow/python/data/experimental/service:_pywrap_snapshot_utils",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "multi_device_test",
    size = "small",
    srcs = ["multi_device_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:lookup_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "multi_process_cluster",
    srcs = ["multi_process_cluster.py"],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/distribute:multi_process_lib",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "multi_process_cluster_test",
    size = "medium",
    srcs = ["multi_process_cluster_test.py"],
    shard_count = 3,
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_ops",
        "//tensorflow/python/ops:math_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "test_base",
    srcs = ["test_base.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/experimental/service:server_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
    ],
)

tf_py_strict_test(
    name = "worker_tags_test",
    size = "medium",
    srcs = ["worker_tags_test.py"],
    shard_count = 32,
    tags = [
        "no_oss_py313",  # TODO(b/427743382): Fix the timeout in linux_x86_cpu_wheel_py313_np1
    ],
    deps = [
        ":multi_process_cluster",
        ":test_base",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)
