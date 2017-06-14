package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "checker",
    hdrs = ["checker.h"],
    deps = [
        "//tensorflow/tools/tfprof/internal:tfprof_stats",
    ],
)

cc_library(
    name = "internal_checker_runner_dummy",
    srcs = ["internal_checker_runner_dummy.cc"],
    hdrs = ["internal_checker_runner.h"],
    deps = [
        ":checker",
    ],
)

cc_library(
    name = "accelerator_utilization_checker",
    hdrs = ["accelerator_utilization_checker.h"],
    deps = [
        ":checker",
    ],
)

cc_library(
    name = "operation_checker",
    hdrs = ["operation_checker.h"],
    deps = [
        ":checker",
    ],
)

cc_library(
    name = "tfprof_advisor",
    hdrs = ["tfprof_advisor.h"],
    deps = [
        ":accelerator_utilization_checker",
        ":checker",
        ":internal_checker_runner_dummy",
        ":operation_checker",
    ],
)

tf_cc_test(
    name = "tfprof_advisor_test",
    srcs = ["tfprof_advisor_test.cc"],
    deps = [
        ":tfprof_advisor",
        "//tensorflow/tools/tfprof/internal:tfprof_tf_testlib",
    ],
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
