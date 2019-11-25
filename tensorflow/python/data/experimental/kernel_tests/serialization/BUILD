load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "dataset_serialization_test_base",
    srcs = [
        "dataset_serialization_test_base.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:iterator_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "auto_shard_dataset_serialization_test",
    size = "medium",
    srcs = ["auto_shard_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "batch_dataset_serialization_test",
    size = "medium",
    srcs = ["batch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "cache_dataset_serialization_test",
    size = "small",
    srcs = ["cache_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "checkpoint_input_pipeline_hook_test",
    size = "small",
    srcs = ["checkpoint_input_pipeline_hook_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
    ],
    tags = [
        "no_pip",
        "no_windows",
        "notsan",
    ],
)

tf_py_test(
    name = "choose_fastest_branch_dataset_serialization_test",
    size = "medium",
    srcs = ["choose_fastest_branch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/experimental/ops:optimization",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "choose_fastest_dataset_serialization_test",
    size = "small",
    srcs = ["choose_fastest_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:optimization",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "concatenate_dataset_serialization_test",
    size = "small",
    srcs = ["concatenate_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "csv_dataset_serialization_test",
    size = "small",
    srcs = ["csv_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/ops:readers",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "dataset_constructor_serialization_test",
    size = "medium",
    srcs = ["dataset_constructor_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "filter_dataset_serialization_test",
    size = "medium",
    srcs = ["filter_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "fixed_length_record_dataset_serialization_test",
    size = "medium",
    srcs = ["fixed_length_record_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/kernel_tests:reader_dataset_ops_test_base",
        "//tensorflow/python/data/ops:readers",
    ],
    shard_count = 4,
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "flat_map_dataset_serialization_test",
    size = "medium",
    srcs = ["flat_map_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "group_by_reducer_serialization_test",
    size = "medium",
    srcs = ["group_by_reducer_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "group_by_window_serialization_test",
    size = "medium",
    srcs = ["group_by_window_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "ignore_errors_serialization_test",
    size = "small",
    srcs = ["ignore_errors_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:error_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "interleave_dataset_serialization_test",
    size = "medium",
    srcs = ["interleave_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "map_and_batch_dataset_serialization_test",
    size = "medium",
    srcs = ["map_and_batch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "map_dataset_serialization_test",
    size = "medium",
    srcs = ["map_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "matching_files_dataset_serialization_test",
    size = "small",
    srcs = ["matching_files_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:matching_files",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_windows",
    ],
)

tf_py_test(
    name = "optimize_dataset_serialization_test",
    size = "small",
    srcs = ["optimize_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:optimization",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "rebatch_dataset_serialization_test",
    size = "small",
    srcs = ["rebatch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "padded_batch_dataset_serialization_test",
    size = "medium",
    srcs = ["padded_batch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "parallel_interleave_dataset_serialization_test",
    size = "medium",
    srcs = ["parallel_interleave_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "parallel_map_dataset_serialization_test",
    size = "medium",
    srcs = ["parallel_map_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "parse_example_dataset_serialization_test",
    size = "medium",
    srcs = ["parse_example_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/kernel_tests:reader_dataset_ops_test_base",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "prefetch_dataset_serialization_test",
    size = "small",
    srcs = ["prefetch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "range_dataset_serialization_test",
    size = "small",
    srcs = ["range_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "sample_from_datasets_serialization_test",
    size = "medium",
    srcs = ["sample_from_datasets_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "scan_dataset_serialization_test",
    size = "small",
    srcs = ["scan_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:scan_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "sequence_dataset_serialization_test",
    size = "medium",
    srcs = ["sequence_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "serialization_integration_test",
    size = "small",
    srcs = ["serialization_integration_test.py"],
    additional_deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
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
    name = "shard_dataset_serialization_test",
    size = "medium",
    srcs = ["shard_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "shuffle_and_repeat_dataset_serialization_test",
    size = "medium",
    srcs = ["shuffle_and_repeat_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:shuffle_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "shuffle_dataset_serialization_test",
    size = "medium",
    srcs = ["shuffle_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "sql_dataset_serialization_test",
    size = "small",
    srcs = ["sql_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/data/experimental/kernel_tests:sql_dataset_test_base",
        "//tensorflow/python/data/experimental/ops:readers",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "stats_dataset_serialization_test",
    size = "medium",
    srcs = ["stats_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/ops:stats_aggregator",
        "//tensorflow/python/data/experimental/ops:stats_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "take_while_dataset_serialization_test",
    size = "medium",
    srcs = ["take_while_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:take_while_ops",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "textline_dataset_serialization_test",
    size = "medium",
    srcs = ["textline_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/kernel_tests:reader_dataset_ops_test_base",
        "//tensorflow/python/data/ops:readers",
    ],
    shard_count = 4,
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "tf_record_dataset_serialization_test",
    size = "medium",
    srcs = ["tf_record_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/kernel_tests:reader_dataset_ops_test_base",
        "//tensorflow/python/data/ops:readers",
    ],
    shard_count = 4,
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "unbatch_dataset_serialization_test",
    size = "medium",
    srcs = ["unbatch_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "unique_dataset_serialization_test",
    size = "small",
    srcs = ["unique_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:unique",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "zip_dataset_serialization_test",
    size = "small",
    srcs = ["zip_dataset_serialization_test.py"],
    additional_deps = [
        ":dataset_serialization_test_base",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
    tags = [
        "no_oss",
        "no_pip",
        "no_windows",
    ],
)
