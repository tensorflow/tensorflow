# Description:
#   Contains experimental kernels for datasets and iterators.
package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
)

tf_kernel_library(
    name = "assert_next_dataset_op",
    srcs = ["assert_next_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "choose_fastest_branch_dataset_op",
    srcs = ["choose_fastest_branch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//tensorflow/core/kernels/data:take_dataset_op",
    ],
)

tf_kernel_library(
    name = "csv_dataset_op",
    srcs = ["csv_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "dense_to_sparse_batch_dataset_op",
    srcs = ["dense_to_sparse_batch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "directed_interleave_dataset_op",
    srcs = ["directed_interleave_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "auto_shard_dataset_op",
    srcs = ["auto_shard_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/optimizers/data:auto_shard",
        "//tensorflow/core/kernels/data:graph_rewrite_dataset",
    ],
)

tf_kernel_library(
    name = "group_by_reducer_dataset_op",
    srcs = ["group_by_reducer_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "group_by_window_dataset_op",
    srcs = ["group_by_window_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//tensorflow/core/kernels/data:window_dataset",
    ],
)

tf_kernel_library(
    name = "ignore_errors_dataset_op",
    srcs = ["ignore_errors_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "indexed_dataset_op",
    srcs = ["indexed_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "lmdb_dataset_op",
    srcs = ["lmdb_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
        "@lmdb",
    ],
)

tf_kernel_library(
    name = "map_and_batch_dataset_op",
    srcs = ["map_and_batch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core/kernels:inplace_ops",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//tensorflow/core/kernels/data:stats_utils",
    ],
)

tf_kernel_library(
    name = "matching_files_dataset_op",
    srcs = ["matching_files_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:dataset",
    ],
)

tf_kernel_library(
    name = "choose_fastest_dataset_op",
    srcs = ["choose_fastest_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:dataset",
    ],
)

tf_kernel_library(
    name = "non_serializable_dataset_op",
    srcs = ["non_serializable_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "numa_map_and_batch_dataset_op",
    srcs = ["numa_map_and_batch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core/kernels:inplace_ops",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
        "@com_google_absl//absl/memory",
    ],
)

tf_kernel_library(
    name = "parallel_interleave_dataset_op",
    srcs = ["parallel_interleave_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "parse_example_dataset_op",
    srcs = ["parse_example_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core/kernels/data:parallel_map_iterator",
        "//tensorflow/core/kernels/data:stats_utils",
    ],
)

tf_kernel_library(
    name = "prefetching_kernels",
    srcs = ["prefetching_kernels.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "random_dataset_op",
    srcs = ["random_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "rebatch_dataset_op",
    srcs = ["rebatch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/optimizers/data:rebatch",
        "//tensorflow/core/kernels/data:graph_rewrite_dataset",
    ],
)

tf_kernel_library(
    name = "sampling_dataset_op",
    srcs = ["sampling_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "scan_dataset_op",
    srcs = ["scan_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "set_stats_aggregator_dataset_op",
    srcs = ["set_stats_aggregator_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:stats_utils",
    ],
)

tf_kernel_library(
    name = "sleep_dataset_op",
    srcs = ["sleep_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
    ],
)

tf_kernel_library(
    name = "sliding_window_dataset_op",
    srcs = ["sliding_window_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "sql_dataset_op",
    srcs = [
        "sql_dataset_op.cc",
    ],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data/experimental/sql",
    ],
)

tf_kernel_library(
    name = "stats_aggregator_ops",
    srcs = ["stats_aggregator_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:summary_interface",
    ],
)

tf_kernel_library(
    name = "stats_dataset_ops",
    srcs = ["stats_dataset_ops.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "take_while_dataset_op",
    srcs = ["take_while_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data:captured_function",
        "//tensorflow/core/kernels/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "threadpool_dataset_op",
    srcs = ["threadpool_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels/data:dataset_utils",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "to_tf_record_op",
    srcs = ["to_tf_record_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/kernels/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "unbatch_dataset_op",
    srcs = ["unbatch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "unique_dataset_op",
    srcs = ["unique_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "dataset_kernels",
    deps = [
        ":assert_next_dataset_op",
        ":auto_shard_dataset_op",
        ":choose_fastest_branch_dataset_op",
        ":choose_fastest_dataset_op",
        ":csv_dataset_op",
        ":dense_to_sparse_batch_dataset_op",
        ":directed_interleave_dataset_op",
        ":group_by_reducer_dataset_op",
        ":group_by_window_dataset_op",
        ":ignore_errors_dataset_op",
        ":indexed_dataset_op",
        ":lmdb_dataset_op",
        ":map_and_batch_dataset_op",
        ":matching_files_dataset_op",
        ":non_serializable_dataset_op",
        ":numa_map_and_batch_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parse_example_dataset_op",
        ":prefetching_kernels",
        ":random_dataset_op",
        ":rebatch_dataset_op",
        ":sampling_dataset_op",
        ":scan_dataset_op",
        ":set_stats_aggregator_dataset_op",
        ":sleep_dataset_op",
        ":sliding_window_dataset_op",
        ":sql_dataset_op",
        ":stats_aggregator_ops",
        ":stats_dataset_ops",
        ":take_while_dataset_op",
        ":threadpool_dataset_op",
        ":to_tf_record_op",
        ":unbatch_dataset_op",
        ":unique_dataset_op",
    ],
)
