# Description: Utilities.

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "periodic_function_dynamic",
    srcs = ["periodic_function.cc"],
    hdrs = ["periodic_function.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "periodic_function",
    visibility = ["//visibility:public"],
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
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        ":periodic_function_dynamic",
        "//tensorflow/core:lib",
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

cc_library(
    name = "fake_clock_env",
    testonly = 1,
    srcs = ["fake_clock_env.cc"],
    hdrs = ["fake_clock_env.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)
