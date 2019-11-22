load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

tf_py_test(
    name = "assert_next_test",
    size = "small",
    srcs = ["assert_next_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "auto_shard_dataset_test",
    size = "medium",
    srcs = ["auto_shard_dataset_test.py"],
    additional_deps = [
        ":reader_dataset_ops_test_base",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
    tags = [
        "no_pip",
    ],
)

tf_py_test(
    name = "bucket_by_sequence_length_test",
    size = "medium",
    srcs = ["bucket_by_sequence_length_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "cardinality_test",
    srcs = ["cardinality_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

cuda_py_test(
    name = "copy_to_device_test",
    size = "small",
    srcs = ["copy_to_device_test.py"],
    additional_deps = [
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/compat:compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
    tags = ["no_windows_gpu"],
)

tf_py_test(
    name = "counter_test",
    size = "small",
    srcs = ["counter_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/experimental/ops:counter",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "csv_dataset_test",
    size = "medium",
    srcs = ["csv_dataset_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/data/experimental/ops:error_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "dense_to_sparse_batch_test",
    srcs = ["dense_to_sparse_batch_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "dense_to_ragged_batch_test",
    srcs = ["dense_to_ragged_batch_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "directed_interleave_dataset_test",
    size = "medium",
    srcs = ["directed_interleave_dataset_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:random_seed",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "get_single_element_test",
    size = "small",
    srcs = ["get_single_element_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:get_single_element",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "group_by_reducer_test",
    size = "medium",
    srcs = ["group_by_reducer_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "group_by_window_test",
    size = "medium",
    srcs = ["group_by_window_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "ignore_errors_test",
    srcs = ["ignore_errors_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:error_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "make_batched_features_dataset_test",
    size = "medium",
    srcs = ["make_batched_features_dataset_test.py"],
    additional_deps = [
        ":reader_dataset_ops_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "make_csv_dataset_test",
    size = "medium",
    srcs = ["make_csv_dataset_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/util:nest",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "make_tf_record_dataset_test",
    size = "medium",
    srcs = ["make_tf_record_dataset_test.py"],
    additional_deps = [
        ":reader_dataset_ops_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/util:nest",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "map_and_batch_test",
    size = "medium",
    srcs = ["map_and_batch_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "map_defun_op_test",
    size = "small",
    srcs = ["map_defun_op_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:map_defun",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "matching_files_test",
    size = "small",
    srcs = ["matching_files_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:matching_files",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "model_dataset_test",
    size = "small",
    srcs = ["model_dataset_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "non_serializable_test",
    size = "small",
    srcs = ["non_serializable_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "optimize_dataset_test",
    size = "medium",
    srcs = ["optimize_dataset_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/experimental/ops:optimization_options",
        "//tensorflow/python/data/experimental/ops:scan_ops",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "override_threadpool_test",
    size = "small",
    srcs = ["override_threadpool_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:script_ops",
        "//tensorflow/python/data/experimental/ops:threadpool",
        "//tensorflow/python/data/experimental/ops:unique",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "parallel_interleave_test",
    size = "medium",
    srcs = ["parallel_interleave_test.py"],
    additional_deps = [
        "@six_archive//:six",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "parse_example_dataset_test",
    size = "small",
    srcs = ["parse_example_dataset_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:parsing_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
    ],
)

cuda_py_test(
    name = "prefetch_to_device_test",
    size = "small",
    srcs = ["prefetch_to_device_test.py"],
    additional_deps = [
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = ["no_windows_gpu"],
)

tf_py_test(
    name = "prefetch_with_slack_test",
    size = "small",
    srcs = ["prefetch_with_slack_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
    ],
)

py_library(
    name = "reader_dataset_ops_test_base",
    srcs = [
        "reader_dataset_ops_test_base.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

tf_py_test(
    name = "rebatch_dataset_test",
    size = "small",
    srcs = ["rebatch_dataset_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/ops/ragged:ragged_tensor",
    ],
)

tf_py_test(
    name = "rejection_resample_test",
    size = "medium",
    srcs = ["rejection_resample_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:resampling",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    shard_count = 5,
    tags = [
        "noasan",
        "optonly",
    ],
)

tf_py_test(
    name = "replicate_test",
    srcs = ["replicate_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    grpc_enabled = True,
    tags = ["no_oss"],
)

tf_py_test(
    name = "replicate_cluster_test",
    srcs = ["replicate_cluster_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    grpc_enabled = True,
    tags = ["no_oss"],
)

cuda_py_test(
    name = "scan_test",
    size = "small",
    srcs = ["scan_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:test_ops",
        "//tensorflow/python/data/experimental/ops:scan_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "shuffle_and_repeat_test",
    size = "medium",
    srcs = ["shuffle_and_repeat_test.py"],
    additional_deps = [
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/ops:shuffle_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_pip",
        "optonly",
    ],
)

tf_py_test(
    name = "sleep_test",
    srcs = ["sleep_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

py_library(
    name = "sql_dataset_test_base",
    srcs = ["sql_dataset_test_base.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow/python/data/experimental/kernel_tests:__pkg__",
        "//tensorflow/python/data/experimental/kernel_tests/serialization:__pkg__",
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "@org_sqlite//:python",
    ],
)

tf_py_test(
    name = "sql_dataset_test",
    size = "medium",
    srcs = ["sql_dataset_test.py"],
    additional_deps = [
        ":sql_dataset_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "snapshot_test",
    size = "medium",
    srcs = ["snapshot_test.py"],
    additional_deps = [
        ":reader_dataset_ops_test_base",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/experimental/ops:snapshot",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
    ],
    shard_count = 10,
)

tf_py_test(
    name = "stats_dataset_ops_test",
    size = "large",
    srcs = ["stats_dataset_ops_test.py"],
    additional_deps = [
        ":reader_dataset_ops_test_base",
        ":stats_dataset_test_base",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:stats_aggregator",
        "//tensorflow/python/data/experimental/ops:stats_ops",
        "//tensorflow/python/data/experimental/ops:stats_options",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_pip",
    ],
)

py_library(
    name = "stats_dataset_test_base",
    srcs = ["stats_dataset_test_base.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/kernel_tests:test_base",
    ],
)

tf_py_test(
    name = "take_while_test",
    size = "small",
    srcs = ["take_while_test.py"],
    additional_deps = [
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/experimental/ops:take_while_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
    ],
)

tf_py_test(
    name = "tf_record_writer_test",
    size = "small",
    srcs = ["tf_record_writer_test.py"],
    additional_deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:writers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

tf_py_test(
    name = "unique_test",
    size = "small",
    srcs = ["unique_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/data/experimental/ops:unique",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = ["no_pip"],
)

tf_py_test(
    name = "variant_test",
    srcs = ["variant_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

cuda_py_test(
    name = "wrap_unwrap_test",
    size = "small",
    srcs = ["wrap_unwrap_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)
