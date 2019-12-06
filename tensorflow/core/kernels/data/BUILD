# Description:
#   OpKernels for tf.data

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_kernel_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(mrry): Remove this empty forwarding library.
cc_library(
    name = "dataset",
    srcs = [],
    hdrs = ["dataset.h"],
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "dataset_test_base",
    testonly = 1,
    srcs = ["dataset_test_base.cc"],
    hdrs = ["dataset_test_base.h"],
    deps = [
        ":batch_dataset_op",
        ":dataset_utils",
        ":iterator_ops",
        ":map_dataset_op",
        ":name_utils",
        ":range_dataset_op",
        ":take_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:ops_testutil",
    ],
)

cc_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.cc"],
    hdrs = ["dataset_utils.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cc_test(
    name = "dataset_utils_test",
    srcs = ["dataset_utils_test.cc"],
    deps = [
        ":dataset_utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "name_utils",
    srcs = ["name_utils.cc"],
    hdrs = ["name_utils.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "name_utils_test",
    srcs = ["name_utils_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":name_utils",
        ":parallel_interleave_dataset_op",
        ":range_dataset_op",
        ":shuffle_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "rewrite_utils",
    srcs = ["rewrite_utils.cc"],
    hdrs = ["rewrite_utils.h"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/optimizers/data",
        "//tensorflow/core/grappler/optimizers/data:function_utils",
        "//tensorflow/core/grappler/optimizers/data:graph_utils",
    ],
)

cc_library(
    name = "stats_utils",
    srcs = ["stats_utils.cc"],
    hdrs = ["stats_utils.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    deps = [
        ":dataset_utils",
        ":single_threaded_executor",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:variable_ops",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "single_threaded_executor",
    srcs = ["single_threaded_executor.cc"],
    hdrs = ["single_threaded_executor.h"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "single_threaded_executor_test",
    srcs = ["single_threaded_executor_test.cc"],
    deps = [
        ":single_threaded_executor",
        "//tensorflow/core:bitwise_ops_op_lib",
        "//tensorflow/core:control_flow_ops_op_lib",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:math_ops_op_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:random_ops_op_lib",
        "//tensorflow/core:spectral_ops_op_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:array",
        "//tensorflow/core/kernels:control_flow_ops",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/kernels:random_ops",
        "//tensorflow/core/kernels:state",
    ],
)

cc_library(
    name = "unbounded_thread_pool",
    srcs = ["unbounded_thread_pool.cc"],
    hdrs = ["unbounded_thread_pool.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "unbounded_thread_pool_test",
    srcs = ["unbounded_thread_pool_test.cc"],
    deps = [
        ":unbounded_thread_pool",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "window_dataset",
    srcs = ["window_dataset.cc"],
    hdrs = ["window_dataset.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "batch_dataset_op",
    srcs = ["batch_dataset_op.cc"],
    hdrs = ["batch_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "batch_dataset_op_test",
    size = "small",
    srcs = ["batch_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "shard_dataset_op",
    srcs = ["shard_dataset_op.cc"],
    hdrs = ["shard_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "shard_dataset_op_test",
    size = "small",
    srcs = ["shard_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        ":shard_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "window_dataset_op",
    srcs = ["window_dataset_op.cc"],
    hdrs = ["window_dataset_op.h"],
    deps = [
        ":name_utils",
        ":window_dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "window_dataset_op_test",
    size = "small",
    srcs = ["window_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        ":window_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "padded_batch_dataset_op",
    srcs = ["padded_batch_dataset_op.cc"],
    hdrs = ["padded_batch_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "padded_batch_dataset_op_test",
    size = "small",
    srcs = ["padded_batch_dataset_op_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":padded_batch_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "filter_dataset_op",
    srcs = ["filter_dataset_op.cc"],
    hdrs = ["filter_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "filter_dataset_op_test",
    size = "small",
    srcs = ["filter_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":filter_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:unique_op",
    ],
)

tf_kernel_library(
    name = "map_dataset_op",
    srcs = ["map_dataset_op.cc"],
    hdrs = ["map_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "map_dataset_op_test",
    size = "small",
    srcs = ["map_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":map_dataset_op",
        ":range_dataset_op",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "parallel_map_dataset_op",
    srcs = ["parallel_map_dataset_op.cc"],
    hdrs = ["parallel_map_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "parallel_map_dataset_op_test",
    size = "small",
    srcs = ["parallel_map_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":parallel_map_dataset_op",
        ":range_dataset_op",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "generator_dataset_op",
    srcs = ["generator_dataset_op.cc"],
    hdrs = ["generator_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
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
    hdrs = ["flat_map_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "flat_map_dataset_op_test",
    size = "small",
    srcs = ["flat_map_dataset_op_test.cc"],
    deps = [
        ":captured_function",
        ":dataset_test_base",
        ":dataset_utils",
        ":flat_map_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "interleave_dataset_op",
    srcs = ["interleave_dataset_op.cc"],
    hdrs = ["interleave_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "interleave_dataset_op_test",
    size = "small",
    srcs = ["interleave_dataset_op_test.cc"],
    deps = [
        ":captured_function",
        ":dataset_test_base",
        ":dataset_utils",
        ":interleave_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "parallel_interleave_dataset_op",
    srcs = ["parallel_interleave_dataset_op.cc"],
    hdrs = ["parallel_interleave_dataset_op.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":name_utils",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "parallel_interleave_dataset_op_test",
    size = "small",
    srcs = ["parallel_interleave_dataset_op_test.cc"],
    deps = [
        ":captured_function",
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":parallel_interleave_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

cc_library(
    name = "prefetch_autotuner",
    srcs = ["prefetch_autotuner.cc"],
    hdrs = ["prefetch_autotuner.h"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "prefetch_autotuner_test",
    srcs = ["prefetch_autotuner_test.cc"],
    deps = [
        ":prefetch_autotuner",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_kernel_library(
    name = "prefetch_dataset_op",
    srcs = ["prefetch_dataset_op.cc"],
    hdrs = ["prefetch_dataset_op.h"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":prefetch_autotuner",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "prefetch_dataset_op_test",
    size = "small",
    srcs = ["prefetch_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":prefetch_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "repeat_dataset_op",
    srcs = ["repeat_dataset_op.cc"],
    hdrs = ["repeat_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "repeat_dataset_op_test",
    size = "small",
    srcs = ["repeat_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":repeat_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "take_dataset_op",
    srcs = ["take_dataset_op.cc"],
    hdrs = ["take_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "take_dataset_op_test",
    size = "small",
    srcs = ["take_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":take_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "skip_dataset_op",
    srcs = ["skip_dataset_op.cc"],
    hdrs = ["skip_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "skip_dataset_op_test",
    size = "small",
    srcs = ["skip_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":skip_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "range_dataset_op",
    srcs = ["range_dataset_op.cc"],
    hdrs = ["range_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "range_dataset_op_test",
    size = "small",
    srcs = ["range_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "shuffle_dataset_op",
    srcs = ["shuffle_dataset_op.cc"],
    hdrs = ["shuffle_dataset_op.h"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        ":random_seed_ops",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "shuffle_dataset_op_test",
    size = "small",
    srcs = ["shuffle_dataset_op_test.cc"],
    deps = [
        "shuffle_dataset_op",
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "sparse_tensor_slice_dataset_op",
    srcs = ["sparse_tensor_slice_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "sparse_tensor_slice_dataset_op_test",
    size = "small",
    srcs = ["sparse_tensor_slice_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":sparse_tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "tensor_dataset_op",
    srcs = ["tensor_dataset_op.cc"],
    hdrs = ["tensor_dataset_op.h"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
    ],
)

tf_cc_test(
    name = "tensor_dataset_op_test",
    size = "small",
    srcs = ["tensor_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":tensor_dataset_op",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "tensor_slice_dataset_op",
    srcs = ["tensor_slice_dataset_op.cc"],
    hdrs = ["tensor_slice_dataset_op.h"],
    deps = [
        ":dataset_utils",
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
    ],
)

tf_cc_test(
    name = "tensor_slice_dataset_op_test",
    size = "small",
    srcs = ["tensor_slice_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "zip_dataset_op",
    srcs = ["zip_dataset_op.cc"],
    hdrs = ["zip_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "zip_dataset_op_test",
    size = "small",
    srcs = ["zip_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        ":zip_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "concatenate_dataset_op",
    srcs = ["concatenate_dataset_op.cc"],
    hdrs = ["concatenate_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "concatenate_dataset_op_test",
    size = "small",
    srcs = ["concatenate_dataset_op_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "text_line_dataset_op",
    srcs = ["text_line_dataset_op.cc"],
    hdrs = ["text_line_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "text_line_dataset_op_test",
    size = "small",
    srcs = ["text_line_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":text_line_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "fixed_length_record_dataset_op",
    srcs = ["fixed_length_record_dataset_op.cc"],
    hdrs = ["fixed_length_record_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "fixed_length_record_dataset_op_test",
    size = "small",
    srcs = ["fixed_length_record_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":fixed_length_record_dataset_op",
        ":iterator_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "tf_record_dataset_op",
    srcs = ["tf_record_dataset_op.cc"],
    hdrs = ["tf_record_dataset_op.h"],
    deps = [
        ":name_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "tf_record_dataset_op_test",
    size = "small",
    srcs = ["tf_record_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":tf_record_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "iterator_ops",
    srcs = ["iterator_ops.cc"],
    hdrs = ["iterator_ops.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":optional_ops",
        ":unbounded_thread_pool",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/kernels:ops_util",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "reduce_dataset_op_test",
    size = "small",
    srcs = ["reduce_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":range_dataset_op",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "multi_device_iterator_ops",
    srcs = ["multi_device_iterator_ops.cc"],
    deps = [
        ":dataset_utils",
        ":iterator_ops",
        ":unbounded_thread_pool",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "optional_ops",
    srcs = ["optional_ops.cc"],
    hdrs = ["optional_ops.h"],
    gpu_srcs = [
        "optional_ops.cu.cc",
        "optional_ops.h",
    ],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

tf_kernel_library(
    name = "cache_dataset_ops",
    srcs = ["cache_dataset_ops.cc"],
    hdrs = ["cache_dataset_ops.h"],
    deps = [
        ":cache_ops",
        ":name_utils",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

tf_cc_test(
    name = "cache_dataset_ops_test",
    srcs = ["cache_dataset_ops_test.cc"],
    deps = [
        ":cache_dataset_ops",
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:ptr_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "optimize_dataset_op",
    srcs = ["optimize_dataset_op.cc"],
    hdrs = ["optimize_dataset_op.h"],
    deps = [
        ":dataset_utils",
        ":rewrite_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "optimize_dataset_op_test",
    size = "small",
    srcs = ["optimize_dataset_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":iterator_ops",
        ":optimize_dataset_op",
        ":range_dataset_op",
        ":take_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_kernel_library(
    name = "model_dataset_op",
    srcs = ["model_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.cc"],
    hdrs = ["dataset_ops.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler/utils:traversal",
    ],
)

tf_kernel_library(
    name = "data",
    deps = [
        ":batch_dataset_op",
        ":cache_dataset_ops",
        ":concatenate_dataset_op",
        ":dataset_ops",
        ":filter_dataset_op",
        ":fixed_length_record_dataset_op",
        ":flat_map_dataset_op",
        ":generator_dataset_op",
        ":interleave_dataset_op",
        ":iterator_ops",
        ":map_dataset_op",
        ":map_defun_op",
        ":model_dataset_op",
        ":multi_device_iterator_ops",
        ":optimize_dataset_op",
        ":optional_ops",
        ":padded_batch_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parallel_map_dataset_op",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        ":repeat_dataset_op",
        ":shard_dataset_op",
        ":shuffle_dataset_op",
        ":skip_dataset_op",
        ":sparse_tensor_slice_dataset_op",
        ":take_dataset_op",
        ":tensor_dataset_op",
        ":tensor_slice_dataset_op",
        ":text_line_dataset_op",
        ":tf_record_dataset_op",
        ":window_dataset_op",
        ":zip_dataset_op",
        "//tensorflow/core:array_ops_op_lib",
        "//tensorflow/core:nn_ops_op_lib",
        "//tensorflow/core/kernels/data/experimental:dataset_kernels",
    ],
)

tf_kernel_library(
    name = "map_defun_op",
    srcs = ["map_defun_op.cc"],
    hdrs = ["map_defun_op.h"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "map_defun_op_test",
    size = "small",
    srcs = ["map_defun_op_test.cc"],
    deps = [
        ":dataset_test_base",
        ":dataset_utils",
        ":map_defun_op",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "random_seed_ops",
    srcs = ["random_seed_ops.cc"],
    hdrs = ["random_seed_ops.h"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "cache_ops",
    srcs = ["cache_ops.cc"],
    hdrs = ["cache_ops.h"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "android_all_op_kernels",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "dataset_ops*",  # includes grappler dependency, which isn't supported on mobile.
            "optimize_dataset_op.*",  # includes grappler dependency, which isn't supported on mobile.
            "rewrite_utils*",  # includes grappler dependency, which isn't supported on mobile.
            "*test.cc",
            "*test.h",
            "*_test_*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
