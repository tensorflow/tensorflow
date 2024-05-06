# Description:
#   Contains experimental kernels for datasets and iterators.

load(
    "//tensorflow:tensorflow.bzl",
    "if_not_mobile",
    "tf_cc_test",
)
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "portable_all_op_kernels_headers",
    srcs = [
        "threadpool_dataset_op.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        "lookup_ops.cc",
        "take_while_dataset_op.cc",
        "threadpool_dataset_op.cc",
        "unbatch_dataset_op.cc",
        ":portable_all_op_kernels_headers",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

tf_kernel_library(
    name = "assert_cardinality_dataset_op",
    srcs = ["assert_cardinality_dataset_op.cc"],
    hdrs = ["assert_cardinality_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "assert_next_dataset_op",
    srcs = ["assert_next_dataset_op.cc"],
    hdrs = ["assert_next_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
    ],
)

tf_cc_test(
    name = "assert_next_dataset_op_test",
    size = "small",
    srcs = ["assert_next_dataset_op_test.cc"],
    deps = [
        ":assert_next_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:range_dataset_op",
        "//tensorflow/core/kernels/data:take_dataset_op",
    ],
)

tf_kernel_library(
    name = "assert_prev_dataset_op",
    srcs = ["assert_prev_dataset_op.cc"],
    hdrs = ["assert_prev_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:rewrite_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:protobuf",
    ],
)

tf_cc_test(
    name = "assert_prev_dataset_op_test",
    size = "small",
    srcs = ["assert_prev_dataset_op_test.cc"],
    deps = [
        ":assert_prev_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:range_dataset_op",
        "//tensorflow/core/kernels/data:take_dataset_op",
        "//tensorflow/core/kernels/data:tensor_slice_dataset_op",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "auto_shard_dataset_op",
    srcs = ["auto_shard_dataset_op.cc"],
    hdrs = ["auto_shard_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:rewrite_utils",
        "//tensorflow/core/grappler/optimizers/data:auto_shard",
    ],
)

tf_cc_test(
    name = "auto_shard_dataset_op_test",
    size = "small",
    srcs = ["auto_shard_dataset_op_test.cc"],
    deps = [
        ":auto_shard_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:type_inference",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:range_dataset_op",
        "//tensorflow/core/kernels/data:shard_dataset_op",
        "@eigen_archive//:eigen3",
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
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels/data:take_dataset_op",
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
    ],
)

tf_kernel_library(
    name = "compression_ops",
    srcs = ["compression_ops.cc"],
    hdrs = ["compression_ops.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:compression_utils",
    ],
)

tf_kernel_library(
    name = "compute_batch_size_op",
    srcs = ["compute_batch_size_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler/optimizers/data:graph_utils",
        "//tensorflow/core/platform:regexp",
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
    name = "data_service_dataset_op",
    srcs = ["data_service_dataset_op.cc"],
    hdrs = ["data_service_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:utils",
        "//tensorflow/core/data/service:common",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_proto_cc",
        "//tensorflow/core/data/service/client:common",
        "//tensorflow/core/data/service/client:data_service_client",
        "//tensorflow/core/data/service/client:utils",
        "//tensorflow/core/kernels/data:parallel_map_dataset_op",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_kernel_library(
    name = "distributed_save_op",
    srcs = ["distributed_save_op.cc"],
    hdrs = ["distributed_save_op.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/core/data/service:py_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_kernel_library(
    name = "data_service_ops",
    srcs = ["data_service_ops.cc"],
    hdrs = ["data_service_ops.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/platform:env_time",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/strings",
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
    hdrs = ["directed_interleave_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "directed_interleave_dataset_op_test",
    size = "small",
    srcs = ["directed_interleave_dataset_op_test.cc"],
    deps = [
        ":directed_interleave_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:range_dataset_op",
        "//tensorflow/core/kernels/data:tensor_slice_dataset_op",
    ],
)

tf_kernel_library(
    name = "global_shuffle_dataset_op",
    srcs = ["global_shuffle_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:random_index_shuffle",
        "//tensorflow/core/kernels/data:random_seed_ops",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
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
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels/data:window_dataset",
    ],
)

tf_kernel_library(
    name = "ignore_errors_dataset_op",
    srcs = ["ignore_errors_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "index_flat_map_dataset_op",
    srcs = ["index_flat_map_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

tf_kernel_library(
    name = "save_dataset_op",
    srcs = ["save_dataset_op.cc"],
    hdrs = ["save_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:hash_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:root_dataset",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/kernels/data:iterator_ops",
    ],
)

tf_cc_test(
    name = "save_dataset_op_test",
    size = "small",
    srcs = ["save_dataset_op_test.cc"],
    deps = [
        ":save_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels/data:range_dataset_op",
    ],
)

tf_kernel_library(
    name = "load_dataset_op",
    srcs = ["load_dataset_op.cc"],
    hdrs = ["load_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data:utils",
        "//tensorflow/core/framework:op_requires",
    ],
)

tf_kernel_library(
    name = "list_dataset_op",
    srcs = ["list_dataset_op.cc"],
    hdrs = ["list_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "list_dataset_op_test",
    size = "small",
    srcs = ["list_dataset_op_test.cc"],
    deps = [
        ":list_dataset_op",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "lmdb_dataset_op",
    srcs = ["lmdb_dataset_op.cc"],
    hdrs = ["lmdb_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "lookup_ops",
    srcs = ["lookup_ops.cc"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:root_dataset",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:initializable_lookup_table",
        "//tensorflow/core/kernels:lookup_util",
        "@com_google_absl//absl/strings",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "map_and_batch_dataset_op",
    srcs = ["map_and_batch_dataset_op.cc"],
    hdrs = ["map_and_batch_dataset_op.h"],
    deps = [
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:inplace_ops",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
    ],
)

tf_cc_test(
    name = "map_and_batch_dataset_op_test",
    size = "small",
    srcs = ["map_and_batch_dataset_op_test.cc"],
    deps = [
        ":map_and_batch_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels/data:range_dataset_op",
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
    ],
)

tf_kernel_library(
    name = "non_serializable_dataset_op",
    srcs = ["non_serializable_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "parallel_interleave_dataset_op",
    srcs = ["parallel_interleave_dataset_op.cc"],
    hdrs = ["parallel_interleave_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
    ],
)

tf_cc_test(
    name = "parallel_interleave_dataset_op_test",
    size = "small",
    srcs = ["parallel_interleave_dataset_op_test.cc"],
    deps = [
        ":parallel_interleave_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:tensor_slice_dataset_op",
    ],
)

tf_kernel_library(
    name = "parse_example_dataset_op",
    srcs = ["parse_example_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:ragged_tensor_variant",
        "//tensorflow/core/kernels/data:parallel_map_dataset_op",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
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
    name = "random_access_ops",
    srcs = ["random_access_ops.cc"],
    hdrs = ["random_access_ops.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:finalization_utils",
        "//tensorflow/core/data:unbounded_thread_pool",
        "//tensorflow/core/kernels/data:iterator_ops",
        "//tensorflow/core/platform",
    ],
)

tf_kernel_library(
    name = "random_dataset_op",
    srcs = ["random_dataset_op.cc"],
    hdrs = ["random_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "//tensorflow/core/kernels/data:random_seed_ops",
    ],
)

tf_cc_test(
    name = "random_dataset_op_test",
    size = "small",
    srcs = ["random_dataset_op_test.cc"],
    deps = [
        ":random_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:random_seed_ops",
        "@eigen_archive//:eigen3",
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
        "//tensorflow/core/data:name_utils",
    ],
)

tf_kernel_library(
    name = "sampling_dataset_op",
    srcs = ["sampling_dataset_op.cc"],
    hdrs = ["sampling_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
    ],
)

tf_cc_test(
    name = "sampling_dataset_op_test",
    size = "small",
    srcs = ["sampling_dataset_op_test.cc"],
    deps = [
        ":sampling_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:range_dataset_op",
        "@eigen_archive//:eigen3",
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
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "set_stats_aggregator_dataset_op",
    srcs = ["set_stats_aggregator_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:stats_utils",
    ],
)

tf_kernel_library(
    name = "sleep_dataset_op",
    srcs = ["sleep_dataset_op.cc"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:dataset_utils",
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
    name = "snapshot_dataset_op",
    srcs = ["snapshot_dataset_op.cc"],
    hdrs = ["snapshot_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:hash_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/framework:op_requires",
        "//tensorflow/core/platform:platform_port",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
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
        "//tensorflow/core:lib_internal",
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
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "threadpool_dataset_op",
    srcs = ["threadpool_dataset_op.cc"],
    hdrs = ["threadpool_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "to_tf_record_op",
    srcs = ["to_tf_record_op.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:root_dataset",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "unbatch_dataset_op",
    srcs = ["unbatch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_kernel_library(
    name = "unique_dataset_op",
    srcs = ["unique_dataset_op.cc"],
    hdrs = ["unique_dataset_op.h"],
    deps = [
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "weighted_flat_map_dataset_op",
    srcs = ["weighted_flat_map_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "unique_dataset_op_test",
    size = "small",
    srcs = ["unique_dataset_op_test.cc"],
    deps = [
        ":unique_dataset_op",
        "//tensorflow/core:experimental_dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/kernels/data:tensor_slice_dataset_op",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    # data service kernels depend on GRPC, so we package them separately
    # so that downstream rules can avoid depending on GRPC.
    name = "data_service_kernels",
    deps = [
        ":data_service_dataset_op",
        ":data_service_ops",
        ":distributed_save_op",
    ],
)

tf_kernel_library(
    name = "experimental",
    deps = [
        ":assert_cardinality_dataset_op",
        ":assert_next_dataset_op",
        ":assert_prev_dataset_op",
        ":choose_fastest_branch_dataset_op",
        ":choose_fastest_dataset_op",
        ":compression_ops",
        ":csv_dataset_op",
        ":dense_to_sparse_batch_dataset_op",
        ":directed_interleave_dataset_op",
        ":global_shuffle_dataset_op",
        ":group_by_reducer_dataset_op",
        ":group_by_window_dataset_op",
        ":ignore_errors_dataset_op",
        ":index_flat_map_dataset_op",
        ":list_dataset_op",
        ":load_dataset_op",
        ":lookup_ops",
        ":map_and_batch_dataset_op",
        ":matching_files_dataset_op",
        ":non_serializable_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parse_example_dataset_op",
        ":prefetching_kernels",
        ":random_access_ops",
        ":random_dataset_op",
        ":rebatch_dataset_op",
        ":sampling_dataset_op",
        ":save_dataset_op",
        ":scan_dataset_op",
        ":set_stats_aggregator_dataset_op",
        ":sleep_dataset_op",
        ":sliding_window_dataset_op",
        ":snapshot_dataset_op",
        ":sql_dataset_op",
        ":stats_aggregator_ops",
        ":stats_dataset_ops",
        ":take_while_dataset_op",
        ":threadpool_dataset_op",
        ":to_tf_record_op",
        ":unbatch_dataset_op",
        ":unique_dataset_op",
        ":weighted_flat_map_dataset_op",
        "//tensorflow/core/data/service/snapshot:list_snapshot_chunks_dataset_op",
        "//tensorflow/core/data/service/snapshot:snapshot_chunk_dataset_op",
    ] + select({
        "//tensorflow:fuchsia": [],
        "//conditions:default": [":lmdb_dataset_op"],
    }) + if_not_mobile([
        # auto_shard_dataset_op uses RewriteDataset method, which
        # is not defined on mobile platforms
        ":auto_shard_dataset_op",
        # compute_batch_size_op depends on grappler, which
        # should not be included on mobile platforms
        ":compute_batch_size_op",
    ]),
)
