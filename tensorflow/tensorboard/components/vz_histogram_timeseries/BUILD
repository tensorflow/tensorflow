package(default_visibility = ["//tensorflow:internal"])

load("@io_bazel_rules_closure//closure:defs.bzl", "webfiles")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")
load("//tensorflow/tensorboard:defs.bzl", "tensorboard_webcomponent_library")

licenses(["notice"])  # Apache 2.0

webfiles(
    name = "vz_histogram_timeseries",
    srcs = [
        "vz-histogram-timeseries.html",
    ],
    path = "/vz-histogram-timeseries",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:d3",
        "@org_polymer",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)

################################################################################
# MARKED FOR DELETION

tensorboard_webcomponent_library(
    name = "legacy",
    srcs = [
        "index.html",
        "vz-histogram-timeseries.html",
        ":legacy_ts",
    ],
    visibility = ["//visibility:public"],
    destdir = "vz-histogram-timeseries",
    deps = [
        "//tensorflow/tensorboard/components:tf_imports",
        "//third_party/javascript/polymer/v1/polymer:lib",
    ],
)

# This is needed: components/BUILD seeks a legacy_ts rule in this package.
tensorboard_ts_library(
    name = "legacy_ts",
    deps_mgmt = "off",
    runtime = "nodejs",
)
