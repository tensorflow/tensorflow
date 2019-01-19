# Description:
#   OpKernels for tf.data

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_kernel_library",
    "tf_cc_test",
)

# TODO(mrry): Remove this empty forwarding library.
cc_library(
    name = "dataset",
    srcs = [],
    hdrs = ["dataset.h"],
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "dataset_utils",
    srcs = ["dataset_utils.cc"],
    hdrs = ["dataset_utils.h"],
    deps = [
        ":captured_function",
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
    name = "captured_function",
    srcs = ["captured_function.cc"],
    hdrs = ["captured_function.h"],
    deps = [
        ":single_threaded_executor",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:variable_ops",
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

cc_library(
    name = "parallel_map_iterator",
    srcs = ["parallel_map_iterator.cc"],
    hdrs = ["parallel_map_iterator.h"],
    deps = [
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
        "//tensorflow/core:metrics",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_kernel_library(
    name = "generator_dataset_op",
    srcs = ["generator_dataset_op.cc"],
    hdrs = ["generator_dataset_op.h"],
    deps = [
        ":captured_function",
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

tf_kernel_library(
    name = "parallel_interleave_dataset_op",
    srcs = ["parallel_interleave_dataset_op.cc"],
    deps = [
        ":captured_function",
        ":dataset_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:metrics",
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
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:metrics",
        "//tensorflow/core:protos_all_cc",
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

tf_kernel_library(
    name = "take_dataset_op",
    srcs = ["take_dataset_op.cc"],
    deps = [
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

tf_kernel_library(
    name = "tensor_dataset_op",
    srcs = ["tensor_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
    ],
)

tf_kernel_library(
    name = "tensor_slice_dataset_op",
    srcs = ["tensor_slice_dataset_op.cc"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
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

tf_kernel_library(
    name = "reader_dataset_ops",
    srcs = ["reader_dataset_ops.cc"],
    deps = [
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
        ":dataset_utils",
        ":optional_ops",
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

tf_kernel_library(
    name = "optimize_dataset_op",
    srcs = ["optimize_dataset_op.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
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
        "//tensorflow/core:metrics",
        "@com_google_absl//absl/memory",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.cc"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
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
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)
