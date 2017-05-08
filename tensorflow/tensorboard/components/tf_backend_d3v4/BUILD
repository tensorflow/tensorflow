package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow/tensorboard:defs.bzl", "tensorboard_ts_library")

licenses(["notice"])  # Apache 2.0

tensorboard_ts_library(
    name = "ts",
    srcs = [
        "backend.ts",
        "behavior.ts",
        "requestManager.ts",
        "router.ts",
        "urlPathHelpers.ts",
    ],
    deps = [
        "//tensorflow/tensorboard/components/vz_sorting_d3v4:ts",
        "//third_party/javascript/node_modules/typescript:es2015.promise",
        "//third_party/javascript/typings/chai",
        "//third_party/javascript/typings/d3_v4:bundle",
        "//third_party/javascript/typings/lodash",
        "//third_party/javascript/typings/mocha",
        "//third_party/javascript/typings/polymer:polymer_without_externs",
        "//third_party/javascript/typings/sinon",
    ],
)

# TODO(dandelion): Add runners for these tests
tensorboard_ts_library(
    name = "tests",
    srcs = [
        "backendTests.ts",
        "behaviorTests.ts",
        "requestManagerTests.ts",
    ],
    deps = [
        ":ts",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(["**"]),
    tags = ["notsan"],
)
