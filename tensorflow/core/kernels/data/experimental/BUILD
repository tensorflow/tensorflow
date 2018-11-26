# Description:
#   Contains experimental kernels for datasets and iterators.
package(default_visibility = ["//tensorflow:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
)

cc_library(
    name = "indexed_dataset_headers",
    hdrs = ["indexed_dataset.h"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "indexed_dataset",
    srcs = [
        "identity_indexed_dataset.cc",
        "indexed_dataset.cc",
    ],
    deps = [
        ":indexed_dataset_headers",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
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
    name = "ignore_errors_dataset_op",
    srcs = ["ignore_errors_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
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
    name = "threadpool_dataset_op",
    srcs = ["threadpool_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "numa_map_and_batch_dataset_op",
    srcs = ["numa_map_and_batch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:inplace_ops",
        "//tensorflow/core/kernels/data:captured_function",
        "@com_google_absl//absl/memory",
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
    name = "assert_next_dataset_op",
    srcs = ["assert_next_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
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
    name = "non_serializable_dataset_op",
    srcs = ["non_serializable_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//third_party/eigen3",
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
    name = "dataset_kernels",
    deps = [
        ":assert_next_dataset_op",
        ":csv_dataset_op",
        ":directed_interleave_dataset_op",
        ":ignore_errors_dataset_op",
        ":indexed_dataset",
        ":lmdb_dataset_op",
        ":matching_files_dataset_op",
        ":non_serializable_dataset_op",
        ":numa_map_and_batch_dataset_op",
        ":prefetching_kernels",
        ":sleep_dataset_op",
        ":threadpool_dataset_op",
        ":unique_dataset_op",
    ],
)
