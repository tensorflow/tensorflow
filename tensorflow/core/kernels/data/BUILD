# Description:
#   OpKernels for tf.data

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "stats_aggregator",
    hdrs = ["stats_aggregator.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "stats_aggregator_ops",
    srcs = ["stats_aggregator_ops.cc"],
    deps = [
        ":stats_aggregator",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "dataset",
    srcs = ["dataset.cc"],
    hdrs = ["dataset.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.cc"],
    hdrs = ["dataset_utils.h"],
    deps = [
        ":captured_function",
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    deps = [
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:proto_text",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/kernels:variable_ops",
    ],
)

cc_library(
    name = "window_dataset",
    srcs = ["window_dataset.cc"],
    hdrs = ["window_dataset.h"],
    deps = [
        ":dataset",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "batch_dataset_op",
    srcs = ["batch_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:batch_util",
    ],
)

tf_kernel_library(
    name = "padded_batch_dataset_op",
    srcs = ["padded_batch_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "dense_to_sparse_batch_dataset_op",
    srcs = ["dense_to_sparse_batch_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "group_by_window_dataset_op",
    srcs = ["group_by_window_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        ":window_dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "filter_dataset_op",
    srcs = ["filter_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "map_dataset_op",
    srcs = ["map_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "map_and_batch_dataset_op",
    srcs = ["map_and_batch_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:inplace_ops",
    ],
)

tf_kernel_library(
    name = "parallel_map_dataset_op",
    srcs = ["parallel_map_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "scan_dataset_op",
    srcs = ["scan_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "flat_map_dataset_op",
    srcs = ["flat_map_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "interleave_dataset_op",
    srcs = ["interleave_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "parallel_interleave_dataset_op",
    srcs = ["parallel_interleave_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "prefetch_dataset_op",
    srcs = ["prefetch_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "repeat_dataset_op",
    srcs = ["repeat_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "take_dataset_op",
    srcs = ["take_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "skip_dataset_op",
    srcs = ["skip_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "ignore_errors_dataset_op",
    srcs = ["ignore_errors_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "stats_dataset_ops",
    srcs = ["stats_dataset_ops.cc"],
    deps = [
        ":dataset",
        ":stats_aggregator",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "random_dataset_op",
    srcs = ["random_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "range_dataset_op",
    srcs = ["range_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "shuffle_dataset_op",
    srcs = ["shuffle_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "sparse_tensor_slice_dataset_op",
    srcs = ["sparse_tensor_slice_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "tensor_dataset_op",
    srcs = ["tensor_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "tensor_slice_dataset_op",
    srcs = ["tensor_slice_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:batch_util",
    ],
)

tf_kernel_library(
    name = "zip_dataset_op",
    srcs = ["zip_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "concatenate_dataset_op",
    srcs = ["concatenate_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "reader_dataset_ops",
    srcs = ["reader_dataset_ops.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "sql_dataset_ops",
    srcs = [
        "sql_dataset_ops.cc",
    ],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels/data/sql",
    ],
)

tf_kernel_library(
    name = "iterator_ops",
    srcs = ["iterator_ops.cc"],
    deps = [
        ":dataset",
        ":stats_aggregator",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "cache_dataset_ops",
    srcs = ["cache_dataset_ops.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

tf_kernel_library(
    name = "unique_dataset_op",
    srcs = ["unique_dataset_op.cc"],
    deps = [
        ":dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    deps = [
        ":batch_dataset_op",
        ":cache_dataset_ops",
        ":concatenate_dataset_op",
        ":dense_to_sparse_batch_dataset_op",
        ":filter_dataset_op",
        ":flat_map_dataset_op",
        ":group_by_window_dataset_op",
        ":ignore_errors_dataset_op",
        ":interleave_dataset_op",
        ":iterator_ops",
        ":map_and_batch_dataset_op",
        ":map_dataset_op",
        ":padded_batch_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parallel_map_dataset_op",
        ":prefetch_dataset_op",
        ":random_dataset_op",
        ":range_dataset_op",
        ":reader_dataset_ops",
        ":repeat_dataset_op",
        ":scan_dataset_op",
        ":shuffle_dataset_op",
        ":skip_dataset_op",
        ":sparse_tensor_slice_dataset_op",
        ":sql_dataset_ops",
        ":stats_aggregator_ops",
        ":stats_dataset_ops",
        ":take_dataset_op",
        ":tensor_dataset_op",
        ":tensor_slice_dataset_op",
        ":unique_dataset_op",
        ":zip_dataset_op",
    ],
)
