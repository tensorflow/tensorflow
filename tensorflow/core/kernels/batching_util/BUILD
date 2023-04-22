# Description: Utilities.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "periodic_function_dynamic",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
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
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "batch_scheduler",
    hdrs = ["batch_scheduler.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
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
    ],
)

cc_library(
    name = "shared_batch_scheduler_hdrs",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler_hdrs",
        ":periodic_function_dynamic",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        ":periodic_function_dynamic",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "shared_batch_scheduler_test",
    srcs = ["shared_batch_scheduler_test.cc"],
    deps = [
        ":fake_clock_env",
        ":shared_batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
        ":concat_split_util",
        ":shared_batch_scheduler",
        ":threadsafe_status",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:thread_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "//tensorflow/core/util:incremental_barrier",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)
