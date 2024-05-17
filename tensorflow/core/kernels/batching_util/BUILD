# Description: Utilities.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "periodic_function_dynamic",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/lib/core:notification",
        "@com_google_absl//absl/functional:any_invocable",
    ],
)

cc_library(
    name = "input_split_metadata",
    srcs = ["input_split_metadata.cc"],
    hdrs = ["input_split_metadata.h"],
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "batch_input_task",
    hdrs = ["batch_input_task.h"],
    deps = [
        ":batch_scheduler_hdrs",
        ":concat_split_util",
        ":input_split_metadata",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/util:incremental_barrier",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "batch_input_task_test",
    srcs = ["batch_input_task_test.cc"],
    deps = [
        ":batch_input_task",
        ":batch_resource_base",
        ":input_split_metadata",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/framework:node_def_proto_cc",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/kernels:batch_kernels",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "periodic_function",
    deps = [
        ":periodic_function_dynamic",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "periodic_function_test",
    srcs = ["periodic_function_test.cc"],
    deps = [
        ":fake_clock_env",
        ":periodic_function_dynamic",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "batch_scheduler_hdrs",
    hdrs = ["batch_scheduler.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/lib/core:notification",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:criticality",
    ],
)

cc_library(
    name = "batch_scheduler",
    srcs = ["batch_scheduler.cc"],
    hdrs = ["batch_scheduler.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:criticality",
    ],
)

cc_library(
    name = "batch_scheduler_utils",
    srcs = ["batch_scheduler_utils.cc"],
    hdrs = ["batch_scheduler_utils.h"],
    deps = [
        "//tensorflow/core:portable_gif_internal",
    ],
)

cc_library(
    name = "threadsafe_status",
    srcs = ["threadsafe_status.cc"],
    hdrs = ["threadsafe_status.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_test(
    name = "batch_scheduler_test",
    srcs = ["batch_scheduler_test.cc"],
    deps = [
        ":batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "batch_scheduler_utils_test",
    srcs = ["batch_scheduler_utils_test.cc"],
    deps = [
        ":batch_scheduler_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "shared_batch_scheduler_hdrs",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_input_task",
        ":batch_scheduler_hdrs",
        ":batch_scheduler_utils",
        ":periodic_function_dynamic",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:notification",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:context_types_hdrs",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:criticality",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_input_task",
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":periodic_function_dynamic",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:context_types_hdrs",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:criticality",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "shared_batch_scheduler_test",
    size = "small",
    srcs = ["shared_batch_scheduler_test.cc"],
    deps = [
        ":batch_scheduler",
        ":fake_clock_env",
        ":shared_batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "adaptive_shared_batch_scheduler",
    hdrs = ["adaptive_shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        ":periodic_function_dynamic",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "adaptive_shared_batch_scheduler_test",
    srcs = ["adaptive_shared_batch_scheduler_test.cc"],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":adaptive_shared_batch_scheduler",
        ":fake_clock_env",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "serial_device_batch_scheduler",
    hdrs = ["serial_device_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        "//tensorflow/core:lib",
    ],
)

tf_cc_test(
    name = "serial_device_batch_scheduler_test",
    srcs = ["serial_device_batch_scheduler_test.cc"],
    tags = [
        "notap",  # b/110374108
    ],
    deps = [
        ":fake_clock_env",
        ":serial_device_batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "basic_batch_scheduler",
    hdrs = ["basic_batch_scheduler.h"],
    deps = [
        ":shared_batch_scheduler",
    ],
)

tf_cc_test(
    name = "basic_batch_scheduler_test",
    srcs = ["basic_batch_scheduler_test.cc"],
    deps = [
        ":basic_batch_scheduler",
        ":batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "basic_batch_scheduler_benchmark",
    srcs = ["basic_batch_scheduler_benchmark_test.cc"],
    tags = [
        "local",
        "manual",
    ],
    deps = [
        ":basic_batch_scheduler",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
    ],
)

tf_cc_test(
    name = "threadsafe_status_test",
    srcs = ["threadsafe_status_test.cc"],
    deps = [
        ":threadsafe_status",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)

cc_library(
    name = "bounded_executor",
    srcs = ["bounded_executor.cc"],
    hdrs = ["bounded_executor.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/platform:threadpool_interface",
        "@com_google_absl//absl/functional:bind_front",
    ],
)

tf_cc_test(
    name = "bounded_executor_test",
    size = "small",
    srcs = ["bounded_executor_test.cc"],
    deps = [
        ":bounded_executor",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:threadpool_interface",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "concat_split_util",
    hdrs = ["concat_split_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:concat_lib",
        "//tensorflow/core/kernels:split_lib",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "batch_resource_base",
    srcs = ["batch_resource_base.cc"],
    hdrs = ["batch_resource_base.h"],
    deps = [
        ":adaptive_shared_batch_scheduler",
        ":batch_scheduler",
        ":batch_scheduler_utils",
        ":concat_split_util",
        ":input_split_metadata",
        ":shared_batch_scheduler",
        ":threadsafe_status",
        ":warmup",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/common_runtime:cost_constants",
        "//tensorflow/core/common_runtime:cost_measurement",
        "//tensorflow/core/common_runtime:cost_measurement_registry",
        "//tensorflow/core/common_runtime:cost_util",
        "//tensorflow/core/common_runtime:request_cost",
        "//tensorflow/core/common_runtime:request_cost_accessor",
        "//tensorflow/core/common_runtime:request_cost_accessor_registry",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/util:incremental_barrier",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:bind_front",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:criticality",
    ],
)

cc_library(
    name = "warmup",
    srcs = ["warmup.cc"],
    hdrs = ["warmup.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

tf_cc_test(
    name = "batch_resource_base_test",
    srcs = ["batch_resource_base_test.cc"],
    deps = [
        ":batch_resource_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:cost_measurement",
        "//tensorflow/core/common_runtime:cost_measurement_registry",
        "//tensorflow/core/common_runtime:no_op_cost_measurement",
        "//tensorflow/core/common_runtime:request_cost",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:criticality",
    ],
)

tf_cc_test(
    name = "input_split_metadata_test",
    srcs = ["input_split_metadata_test.cc"],
    deps = [
        ":input_split_metadata",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)
