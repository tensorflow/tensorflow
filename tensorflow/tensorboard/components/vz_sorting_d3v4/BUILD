package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")

licenses(["notice"])  # Apache 2.0

tensorboard_ts_library(
    name = "ts",
    srcs = ["sorting.ts"],
)

tensorboard_ts_library(
    name = "tests",
    srcs = ["sortingTests.ts"],
    deps = [
        ":ts",
        "//third_party/javascript/typings/chai",
        "//third_party/javascript/typings/mocha",
        "//third_party/javascript/typings/sinon",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
