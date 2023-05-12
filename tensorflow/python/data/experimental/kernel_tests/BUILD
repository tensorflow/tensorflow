# Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_py_strict_test(
    name = "assert_cardinality_test",
    size = "small",
    srcs = ["assert_cardinality_test.py"],
    deps = [
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "assert_next_test",
    size = "small",
    srcs = ["assert_next_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "assert_prev_test",
    size = "small",
    srcs = ["assert_prev_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "auto_shard_dataset_test",
    size = "medium",
    srcs = ["auto_shard_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/experimental/ops:unique",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/lib/io:lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "checkpoint_input_pipeline_hook_test",
    size = "medium",
    srcs = ["checkpoint_input_pipeline_hook_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/estimator:estimator_py",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/training:saver",
        "//tensorflow/python/training:training_util",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "compression_ops_test",
    size = "small",
    srcs = ["compression_ops_test.py"],
    deps = [
        "//tensorflow/python/data/experimental/ops:compression_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "copy_to_device_test",
    size = "small",
    srcs = ["copy_to_device_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "csv_dataset_test",
    size = "medium",
    srcs = ["csv_dataset_test.py"],
    shard_count = 8,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/data/experimental/ops:error_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "dense_to_sparse_batch_test",
    size = "medium",
    srcs = ["dense_to_sparse_batch_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "from_list_test",
    size = "small",
    srcs = ["from_list_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/experimental/ops:from_list",
        "//tensorflow/python/data/experimental/ops:random_access",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "group_by_reducer_test",
    size = "small",
    srcs = ["group_by_reducer_test.py"],
    shard_count = 12,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "index_shuffle_test",
    size = "large",
    srcs = ["index_shuffle_test.py"],
    shard_count = 12,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:shuffle_ops",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:dtypes",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "io_test",
    size = "medium",
    srcs = ["io_test.py"],
    tags = [
        "notap",  # b/272281090
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/experimental/ops:io",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "lookup_ops_test",
    size = "small",
    srcs = ["lookup_ops_test.py"],
    deps = [
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/experimental/ops:lookup_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_py_strict_test(
    name = "make_batched_features_dataset_test",
    size = "small",
    srcs = ["make_batched_features_dataset_test.py"],
    shard_count = 6,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_csv_dataset_test",
    size = "small",
    srcs = ["make_csv_dataset_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_saveable_from_iterator_test",
    size = "small",
    srcs = ["make_saveable_from_iterator_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python/data/experimental/ops:iterator_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/training:saver",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "make_tf_record_dataset_test",
    size = "medium",
    srcs = ["make_tf_record_dataset_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_and_batch_test",
    size = "medium",
    srcs = ["map_and_batch_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:cond",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:pywrap_sanitizers",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/checkpoint",
        "//tensorflow/python/checkpoint:checkpoint_management",
        "//tensorflow/python/data/experimental/ops:batching",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:combinations",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "map_defun_op_test",
    size = "small",
    srcs = ["map_defun_op_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:array_ops_stack",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:map_defun",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_spec",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "matching_files_dataset_test",
    size = "small",
    srcs = ["matching_files_dataset_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:matching_files",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "model_dataset_test",
    size = "small",
    srcs = ["model_dataset_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "non_serializable_test",
    size = "small",
    srcs = ["non_serializable_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "parallel_interleave_test",
    size = "medium",
    srcs = ["parallel_interleave_test.py"],
    shard_count = 8,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:interleave_ops",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "parse_example_dataset_test",
    size = "medium",
    srcs = ["parse_example_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:parsing_ops",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/kernel_tests:tf_record_test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "prefetch_to_device_test",
    size = "small",
    srcs = ["prefetch_to_device_test.py"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "prefetch_with_slack_test",
    size = "small",
    srcs = ["prefetch_with_slack_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "rebatch_dataset_test",
    size = "medium",
    srcs = ["rebatch_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:image_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "replicate_test",
    size = "medium",
    srcs = ["replicate_test.py"],
    grpc_enabled = True,
    tags = ["no_oss"],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:options",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/training:server_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "shuffle_and_repeat_test",
    size = "medium",
    srcs = ["shuffle_and_repeat_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:shuffle_ops",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:random_seed",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sleep_test",
    size = "small",
    srcs = ["sleep_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "sql_dataset_test",
    size = "medium",
    srcs = ["sql_dataset_test.py"],
    shard_count = 4,
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/experimental/ops:readers",
        "//tensorflow/python/data/kernel_tests:checkpoint_test_base",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
        "@org_sqlite//:python",
    ],
)

tf_py_strict_test(
    name = "tf_record_writer_test",
    size = "medium",
    srcs = ["tf_record_writer_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lib",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/experimental/ops:grouping",
        "//tensorflow/python/data/experimental/ops:writers",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/util:compat",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_strict_test(
    name = "variant_test",
    size = "small",
    srcs = ["variant_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "wrap_unwrap_test",
    size = "small",
    srcs = ["wrap_unwrap_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/framework:combinations",
        "//tensorflow/python/framework:ops",
        "@absl_py//absl/testing:parameterized",
    ],
)
