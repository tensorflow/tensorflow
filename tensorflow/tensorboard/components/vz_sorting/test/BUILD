package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_wct_test_suite")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

################################################################################
# MARKED FOR DELETION

tensorboard_wct_test_suite(
    name = "legacy_test",
    size = "medium",
    srcs = ["index.html"],
    deps = [
        "//tensorflow/tensorboard/components/vz_sorting:legacy",
        "//third_party/javascript/polymer/v1/webcomponentsjs:lib",
    ],
)

tensorboard_ts_library(
    name = "legacy_ts",
    testonly = 1,
    srcs = ["sortingTests.ts"],
    deps = [
        "//tensorflow/tensorboard/components:common_deps",
        "//tensorflow/tensorboard/components/vz_sorting:legacy_ts",
    ],
)
