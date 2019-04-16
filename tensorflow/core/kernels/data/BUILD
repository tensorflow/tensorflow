# Description:
#   OpKernels for tf.data

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_kernel_library",
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
        ":dataset_utils",
        ":iterator_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
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
    ],
)

tf_cc_test(
    name = "dataset_utils_test",
    srcs = ["dataset_utils_test.cc"],
    deps = [
        ":dataset_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "batch_dataset_op",
    srcs = ["batch_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "shard_dataset_op",
    srcs = ["shard_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "window_dataset_op",
    srcs = ["window_dataset_op.cc"],
    deps = [
        ":window_dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "padded_batch_dataset_op",
    srcs = ["padded_batch_dataset_op.cc"],
    deps = [
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
        ":dataset_utils",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "filter_by_component_dataset_op",
    srcs = ["filter_by_component_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "map_dataset_op",
    srcs = ["map_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset_utils",
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

cc_library(
    name = "parallel_map_iterator",
    srcs = ["parallel_map_iterator.cc"],
    hdrs = ["parallel_map_iterator.h"],
    deps = [
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_kernel_library(
    name = "parallel_map_dataset_op",
    srcs = ["parallel_map_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":parallel_map_iterator",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
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
        ":dataset_utils",
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
    deps = [
        ":captured_function",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
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
    deps = [
        ":captured_function",
        ":dataset_utils",
        ":stats_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
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
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "prefetch_autotuner_test",
    srcs = ["prefetch_autotuner_test.cc"],
    deps = [
        ":prefetch_autotuner",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_kernel_library(
    name = "prefetch_dataset_op",
    srcs = ["prefetch_dataset_op.cc"],
    hdrs = ["prefetch_dataset_op.h"],
    deps = [
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
    deps = [
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
    deps = [
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
    deps = [
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
    deps = [
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
    deps = [
        ":dataset_utils",
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
        ":dataset_utils",
        ":iterator_ops",
        ":tensor_dataset_op",
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
    name = "tensor_slice_dataset_op",
    srcs = ["tensor_slice_dataset_op.cc"],
    deps = [
        ":dataset_utils",
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
    deps = [
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
    deps = [
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
    name = "reader_dataset_ops",
    srcs = ["reader_dataset_ops.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
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

tf_kernel_library(
    name = "multi_device_iterator_ops",
    srcs = ["multi_device_iterator_ops.cc"],
    deps = [
        ":dataset_utils",
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
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/util/tensor_bundle",
    ],
)

cc_library(
    name = "graph_rewrite_dataset",
    srcs = ["graph_rewrite_dataset.cc"],
    hdrs = ["graph_rewrite_dataset.h"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/grappler/optimizers/data",
        "//tensorflow/core/grappler/optimizers/data:function_utils",
        "//tensorflow/core/grappler/optimizers/data:graph_utils",
    ],
)

tf_kernel_library(
    name = "optimize_dataset_op",
    srcs = ["optimize_dataset_op.cc"],
    deps = [
        ":graph_rewrite_dataset",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
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
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "data",
    deps = [
        ":batch_dataset_op",
        ":cache_dataset_ops",
        ":concatenate_dataset_op",
        ":dataset_ops",
        ":filter_by_component_dataset_op",
        ":filter_dataset_op",
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
        ":reader_dataset_ops",
        ":repeat_dataset_op",
        ":shard_dataset_op",
        ":shuffle_dataset_op",
        ":skip_dataset_op",
        ":sparse_tensor_slice_dataset_op",
        ":take_dataset_op",
        ":tensor_dataset_op",
        ":tensor_slice_dataset_op",
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
    deps = [
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)
