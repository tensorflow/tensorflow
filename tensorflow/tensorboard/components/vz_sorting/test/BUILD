package(
    default_testonly = True,
    default_visibility = ["//tensorflow:internal"],
)

load("//tensorflow/tensorboard:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "test",
    srcs = [
        "sortingTests.ts",
        "tests.html",
    ],
    path = "/vz-sorting/test",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:web_component_tester",
        "//tensorflow/tensorboard/components/vz_sorting",
    ],
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
