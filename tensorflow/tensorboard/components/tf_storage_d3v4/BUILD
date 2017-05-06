package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")

licenses(["notice"])  # Apache 2.0

# TODO(dandelion): Add test runner.

tensorboard_ts_library(
    name = "ts",
    srcs = ["storage.ts"],
    deps = [
        "//tensorflow/tensorboard/components/tf_globals_d3v4:ts",
        "//third_party/javascript/lodash:lodash-module",
        "//third_party/javascript/typings/lodash",
    ],
)

tensorboard_ts_library(
    name = "tests",
    srcs = ["storageTests.ts"],
    deps = [
        ":ts",
        "//tensorflow/tensorboard/components/tf_globals_d3v4:ts",
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
