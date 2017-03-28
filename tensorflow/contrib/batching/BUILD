# Description: Batching scheduling library.

package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

cc_library(
    name = "batch_scheduler",
    hdrs = ["batch_scheduler.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:tensorflow",
    ],
)

cc_test(
    name = "batch_scheduler_test",
    srcs = [
        "batch_scheduler_test.cc",
    ],
    deps = [
        ":batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "shared_batch_scheduler",
    hdrs = ["shared_batch_scheduler.h"],
    deps = [
        ":batch_scheduler",
        "//tensorflow/contrib/batching/util:periodic_function",
        "//tensorflow/core:lib",
    ],
)

cc_test(
    name = "shared_batch_scheduler_test",
    srcs = [
        "shared_batch_scheduler_test.cc",
    ],
    deps = [
        ":shared_batch_scheduler",
        "//tensorflow/contrib/batching/test_util:fake_clock_env",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
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

cc_test(
    name = "basic_batch_scheduler_test",
    srcs = [
        "basic_batch_scheduler_test.cc",
    ],
    deps = [
        ":basic_batch_scheduler",
        ":batch_scheduler",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "basic_batch_scheduler_benchmark",
    srcs = ["basic_batch_scheduler_benchmark.cc"],
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
