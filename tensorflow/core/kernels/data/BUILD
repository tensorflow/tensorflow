# Description:
#   OpKernels for tf.data

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

# Definitions are loaded separately so that copybara can pattern match (and modify) each definition.
load("//tensorflow:tensorflow.default.bzl", "filegroup", "tf_kernel_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_kernel_library(
    name = "batch_dataset_op",
    srcs = ["batch_dataset_op.cc"],
    hdrs = ["batch_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_cc_test(
    name = "batch_dataset_op_test",
    size = "small",
    srcs = ["batch_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":iterator_ops",
        ":range_dataset_op",
        "//tensorflow/core",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:type_inference",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "cache_dataset_ops",
    srcs = ["cache_dataset_ops.cc"],
    hdrs = ["cache_dataset_ops.h"],
    deps = [
        ":cache_ops",
        ":iterator_ops",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/core/util/tensor_bundle:naming",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "cache_dataset_ops_test",
    size = "small",
    srcs = ["cache_dataset_ops_test.cc"],
    deps = [
        ":cache_dataset_ops",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
    ],
)

tf_kernel_library(
    name = "cache_ops",
    srcs = ["cache_ops.cc"],
    hdrs = ["cache_ops.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "concatenate_dataset_op",
    srcs = ["concatenate_dataset_op.cc"],
    hdrs = ["concatenate_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_cc_test(
    name = "concatenate_dataset_op_test",
    size = "small",
    srcs = ["concatenate_dataset_op_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "dataset_ops",
    srcs = ["dataset_ops.cc"],
    hdrs = ["dataset_ops.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:hash_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/grappler:graph_topology_view",
        "//tensorflow/core/grappler/utils:traversal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_kernel_library(
    name = "filter_dataset_op",
    srcs = ["filter_dataset_op.cc"],
    hdrs = ["filter_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
    ],
)

tf_cc_test(
    name = "filter_dataset_op_test",
    size = "small",
    srcs = ["filter_dataset_op_test.cc"],
    deps = [
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
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:unique_op",
    ],
)

tf_kernel_library(
    name = "finalize_dataset_op",
    srcs = ["finalize_dataset_op.cc"],
    hdrs = ["finalize_dataset_op.h"],
    deps = [
        ":model_dataset_op",
        ":optimize_dataset_op",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/kernels/data/experimental:threadpool_dataset_op",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

tf_cc_test(
    name = "finalize_dataset_op_test",
    size = "small",
    srcs = ["finalize_dataset_op_test.cc"],
    deps = [
        ":finalize_dataset_op",
        ":options_dataset_op",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "fixed_length_record_dataset_op",
    srcs = ["fixed_length_record_dataset_op.cc"],
    hdrs = ["fixed_length_record_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:utils",
    ],
)

tf_cc_test(
    name = "fixed_length_record_dataset_op_test",
    size = "small",
    srcs = ["fixed_length_record_dataset_op_test.cc"],
    deps = [
        ":fixed_length_record_dataset_op",
        ":iterator_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "flat_map_dataset_op",
    srcs = ["flat_map_dataset_op.cc"],
    hdrs = ["flat_map_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:flat_map_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "flat_map_dataset_op_test",
    size = "small",
    srcs = ["flat_map_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
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
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "generator_dataset_op",
    srcs = ["generator_dataset_op.cc"],
    hdrs = ["generator_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
    ],
)

tf_kernel_library(
    name = "get_options_op",
    srcs = ["get_options_op.cc"],
    hdrs = ["get_options_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "get_options_op_test",
    size = "small",
    srcs = ["get_options_op_test.cc"],
    deps = [
        ":get_options_op",
        ":options_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "interleave_dataset_op",
    srcs = ["interleave_dataset_op.cc"],
    hdrs = ["interleave_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "interleave_dataset_op_test",
    size = "small",
    srcs = ["interleave_dataset_op_test.cc"],
    deps = [
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
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "iterator_ops",
    srcs = ["iterator_ops.cc"],
    hdrs = ["iterator_ops.h"],
    deps = [
        ":optional_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/activity_watcher",
        "//tensorflow/core/activity_watcher:activity_watcher_utils",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:finalization_utils",
        "//tensorflow/core/data:metric_utils",
        "//tensorflow/core/data:root_dataset",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/data:tf_data_memory_logger",
        "//tensorflow/core/data:tfdataz_metrics",
        "//tensorflow/core/data:unbounded_thread_pool",
        "//tensorflow/core/data:utils",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "iterator_ops_test",
    size = "small",
    srcs = ["iterator_ops_test.cc"],
    deps = [
        ":iterator_ops",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/framework:function_testlib",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/lib/monitoring:cell_reader",
        "//tensorflow/core/lib/monitoring:test_utils",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:statusor",
    ],
)

tf_kernel_library(
    name = "map_dataset_op",
    srcs = ["map_dataset_op.cc"],
    hdrs = ["map_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "map_dataset_op_test",
    size = "small",
    srcs = ["map_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":iterator_ops",
        ":map_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "map_defun_op",
    srcs = ["map_defun_op.cc"],
    hdrs = ["map_defun_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_cc_test(
    name = "map_defun_op_test",
    size = "small",
    srcs = ["map_defun_op_test.cc"],
    deps = [
        ":map_defun_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "model_dataset_op",
    srcs = ["model_dataset_op.cc"],
    hdrs = ["model_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "@com_google_absl//absl/memory",
    ],
)

tf_kernel_library(
    name = "multi_device_iterator_ops",
    srcs = ["multi_device_iterator_ops.cc"],
    deps = [
        ":iterator_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:finalization_utils",
        "//tensorflow/core/data:metric_utils",
        "//tensorflow/core/data:root_dataset",
        "//tensorflow/core/data:tfdataz_metrics",
        "//tensorflow/core/data:unbounded_thread_pool",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:refcount",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ],
)

tf_kernel_library(
    name = "optimize_dataset_op",
    srcs = ["optimize_dataset_op.cc"],
    hdrs = ["optimize_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:rewrite_utils",
        "//tensorflow/core/platform:platform_port",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "optimize_dataset_op_test",
    size = "small",
    srcs = ["optimize_dataset_op_test.cc"],
    deps = [
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
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels/data/experimental:threadpool_dataset_op",
    ],
)

tf_kernel_library(
    name = "optional_ops_util",
    srcs = ["optional_ops_util.cc"],
    hdrs = ["optional_ops_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
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
    visibility = ["//visibility:public"],
    deps = [
        ":optional_ops_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@eigen_archive//:eigen3",
    ],
)

tf_kernel_library(
    name = "options_dataset_op",
    srcs = ["options_dataset_op.cc"],
    hdrs = ["options_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "options_dataset_op_test",
    size = "small",
    srcs = ["options_dataset_op_test.cc"],
    deps = [
        ":options_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "padded_batch_dataset_op",
    srcs = ["padded_batch_dataset_op.cc"],
    hdrs = ["padded_batch_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
    ],
)

tf_cc_test(
    name = "padded_batch_dataset_op_test",
    size = "small",
    srcs = ["padded_batch_dataset_op_test.cc"],
    deps = [
        ":concatenate_dataset_op",
        ":iterator_ops",
        ":padded_batch_dataset_op",
        ":range_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "parallel_batch_dataset_op",
    srcs = ["parallel_batch_dataset_op.cc"],
    hdrs = ["parallel_batch_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/log",
    ],
)

tf_cc_test(
    name = "parallel_batch_dataset_op_test",
    size = "small",
    srcs = ["parallel_batch_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":parallel_batch_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
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
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "parallel_interleave_dataset_op_test",
    size = "small",
    srcs = ["parallel_interleave_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":parallel_interleave_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:identity_op",
    ],
)

tf_kernel_library(
    name = "parallel_map_dataset_op",
    srcs = ["parallel_map_dataset_op.cc"],
    hdrs = ["parallel_map_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/data:unbounded_thread_pool",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_xla//xla/tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "parallel_map_dataset_op_test",
    size = "small",
    srcs = ["parallel_map_dataset_op_test.cc"],
    deps = [
        ":batch_dataset_op",
        ":iterator_ops",
        ":parallel_map_dataset_op",
        ":range_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "@com_google_googletest//:gtest",
    ],
)

tf_kernel_library(
    name = "parallel_filter_dataset_op",
    srcs = ["parallel_filter_dataset_op.cc"],
    hdrs = ["parallel_filter_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "parallel_filter_dataset_op_test",
    size = "small",
    srcs = ["parallel_filter_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":parallel_filter_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
        "//tensorflow/core/kernels:unique_op",
    ],
)

cc_library(
    name = "prefetch_autotuner",
    srcs = ["prefetch_autotuner.cc"],
    hdrs = ["prefetch_autotuner.h"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/data:dataset_utils",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "prefetch_autotuner_test",
    size = "small",
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
        ":prefetch_autotuner",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
    ],
)

tf_cc_test(
    name = "prefetch_dataset_op_test",
    size = "small",
    srcs = ["prefetch_dataset_op_test.cc"],
    deps = [
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
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "random_seed_ops",
    srcs = ["random_seed_ops.cc"],
    hdrs = ["random_seed_ops.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:functional_ops_op_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "range_dataset_op",
    srcs = ["range_dataset_op.cc"],
    hdrs = ["range_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_xla//xla/tsl/platform:types",
    ],
)

tf_cc_test(
    name = "range_dataset_op_test",
    size = "small",
    srcs = ["range_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "reduce_dataset_op",
    srcs = ["reduce_dataset_op.cc"],
    hdrs = ["reduce_dataset_op.h"],
    deps = [
        ":iterator_ops",
        ":prefetch_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:captured_function",
        "//tensorflow/core/data:root_dataset",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "reduce_dataset_op_test",
    size = "small",
    srcs = ["reduce_dataset_op_test.cc"],
    deps = [
        ":range_dataset_op",
        ":reduce_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:stats_utils",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:function_ops",
    ],
)

tf_kernel_library(
    name = "repeat_dataset_op",
    srcs = ["repeat_dataset_op.cc"],
    hdrs = ["repeat_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "repeat_dataset_op_test",
    size = "small",
    srcs = ["repeat_dataset_op_test.cc"],
    deps = [
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
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "rewrite_dataset_op",
    srcs = ["rewrite_dataset_op.cc"],
    hdrs = ["rewrite_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:rewrite_utils",
        "//tensorflow/core/platform:platform_port",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "rewrite_dataset_op_test",
    size = "small",
    srcs = ["rewrite_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":rewrite_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "shard_dataset_op",
    srcs = ["shard_dataset_op.cc"],
    hdrs = ["shard_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "shard_dataset_op_test",
    size = "small",
    srcs = ["shard_dataset_op_test.cc"],
    deps = [
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
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "shuffle_dataset_op",
    srcs = ["shuffle_dataset_op.cc"],
    hdrs = ["shuffle_dataset_op.h"],
    deps = [
        ":random_seed_ops",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "shuffle_dataset_op_test",
    size = "small",
    srcs = ["shuffle_dataset_op_test.cc"],
    deps = [
        "shuffle_dataset_op",
        ":iterator_ops",
        ":range_dataset_op",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "skip_dataset_op",
    srcs = ["skip_dataset_op.cc"],
    hdrs = ["skip_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "skip_dataset_op_test",
    size = "small",
    srcs = ["skip_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":skip_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
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
        ":iterator_ops",
        ":sparse_tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
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
        "//tensorflow/core/data:name_utils",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "take_dataset_op_test",
    size = "small",
    srcs = ["take_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":take_dataset_op",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
    ],
)

tf_kernel_library(
    name = "tensor_dataset_op",
    srcs = ["tensor_dataset_op.cc"],
    hdrs = ["tensor_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "tensor_dataset_op_test",
    size = "small",
    srcs = ["tensor_dataset_op_test.cc"],
    deps = [
        ":tensor_dataset_op",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "tensor_slice_dataset_op",
    srcs = ["tensor_slice_dataset_op.cc"],
    hdrs = ["tensor_slice_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:global_shuffle_utils",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

tf_cc_test(
    name = "tensor_slice_dataset_op_test",
    size = "small",
    srcs = ["tensor_slice_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":tensor_slice_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:dataset_utils",
        "//tensorflow/core/data:serialization_utils",
    ],
)

tf_kernel_library(
    name = "text_line_dataset_op",
    srcs = ["text_line_dataset_op.cc"],
    hdrs = ["text_line_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:utils",
    ],
)

tf_cc_test(
    name = "text_line_dataset_op_test",
    size = "small",
    srcs = ["text_line_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":text_line_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
    ],
)

tf_kernel_library(
    name = "tf_record_dataset_op",
    srcs = ["tf_record_dataset_op.cc"],
    hdrs = ["tf_record_dataset_op.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:utils",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

tf_cc_test(
    name = "tf_record_dataset_op_test",
    size = "small",
    srcs = ["tf_record_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":tf_record_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "window_dataset",
    srcs = ["window_dataset.cc"],
    hdrs = ["window_dataset.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

tf_kernel_library(
    name = "window_dataset_op",
    srcs = ["window_dataset_op.cc"],
    hdrs = ["window_dataset_op.h"],
    deps = [
        ":window_dataset",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "window_dataset_op_test",
    size = "small",
    srcs = ["window_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":window_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:serialization_utils",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_kernel_library(
    name = "zip_dataset_op",
    srcs = ["zip_dataset_op.cc"],
    hdrs = ["zip_dataset_op.h"],
    deps = [
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/data:split_utils",
        "//tensorflow/core/framework:dataset_options_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "zip_dataset_op_test",
    size = "small",
    srcs = ["zip_dataset_op_test.cc"],
    deps = [
        ":iterator_ops",
        ":range_dataset_op",
        ":zip_dataset_op",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:dataset_ops_op_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:dataset_test_base",
        "//tensorflow/core/data:name_utils",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

filegroup(
    name = "portable_all_op_kernels_headers",
    srcs = [
        "//tensorflow/core/data:captured_function.h",
        "//tensorflow/core/data:compression_utils.h",
        "//tensorflow/core/data:dataset_utils.h",
        "//tensorflow/core/data:finalization_utils.h",
        "//tensorflow/core/data:flat_map_utils.h",
        "//tensorflow/core/data:global_shuffle_utils.h",
        "//tensorflow/core/data:metric_utils.h",
        "//tensorflow/core/data:name_utils.h",
        "//tensorflow/core/data:rewrite_utils.h",
        "//tensorflow/core/data:root_dataset.h",
        "//tensorflow/core/data:serialization_utils.h",
        "//tensorflow/core/data:split_utils.h",
        "//tensorflow/core/data:stats_utils.h",
        "//tensorflow/core/data:tf_data_memory_logger.h",
        "//tensorflow/core/data:tfdataz_metrics.h",
        "//tensorflow/core/data:unbounded_thread_pool.h",
        "//tensorflow/core/data:utils.h",
        "//tensorflow/core/kernels/data/experimental:portable_all_op_kernels_headers",
    ] + glob(
        [
            "*.h",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "portable_all_op_kernels",
    srcs = [
        ":portable_all_op_kernels_headers",
        "//tensorflow/core/data:captured_function.cc",
        "//tensorflow/core/data:compression_utils.cc",
        "//tensorflow/core/data:dataset_utils.cc",
        "//tensorflow/core/data:finalization_utils.cc",
        "//tensorflow/core/data:flat_map_utils.cc",
        "//tensorflow/core/data:global_shuffle_utils.cc",
        "//tensorflow/core/data:metric_utils.cc",
        "//tensorflow/core/data:name_utils.cc",
        "//tensorflow/core/data:rewrite_utils.cc",
        "//tensorflow/core/data:root_dataset.cc",
        "//tensorflow/core/data:serialization_utils.cc",
        "//tensorflow/core/data:split_utils.cc",
        "//tensorflow/core/data:stats_utils.cc",
        "//tensorflow/core/data:tf_data_memory_logger.cc",
        "//tensorflow/core/data:tfdataz_metrics.cc",
        "//tensorflow/core/data:unbounded_thread_pool.cc",
        "//tensorflow/core/data:utils.cc",
        "//tensorflow/core/kernels/data/experimental:portable_all_op_kernels",
    ] + glob(
        [
            "*.cc",
        ],
        exclude = [
            "*test.cc",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_kernel_library(
    name = "data",
    visibility = ["//visibility:public"],
    deps = [
        ":batch_dataset_op",
        ":cache_dataset_ops",
        ":concatenate_dataset_op",
        ":dataset_ops",
        ":filter_dataset_op",
        ":finalize_dataset_op",
        ":fixed_length_record_dataset_op",
        ":flat_map_dataset_op",
        ":generator_dataset_op",
        ":get_options_op",
        ":interleave_dataset_op",
        ":iterator_ops",
        ":map_dataset_op",
        ":map_defun_op",
        ":model_dataset_op",
        ":multi_device_iterator_ops",
        ":optimize_dataset_op",
        ":optional_ops",
        ":options_dataset_op",
        ":padded_batch_dataset_op",
        ":parallel_batch_dataset_op",
        ":parallel_filter_dataset_op",
        ":parallel_interleave_dataset_op",
        ":parallel_map_dataset_op",
        ":prefetch_dataset_op",
        ":range_dataset_op",
        ":reduce_dataset_op",
        ":repeat_dataset_op",
        ":rewrite_dataset_op",
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
        "//tensorflow/core/kernels/data/experimental",
    ],
)
