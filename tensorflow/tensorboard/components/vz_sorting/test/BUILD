package(
    default_testonly = True,
    default_visibility = ["//tensorflow/tensorboard:internal"],
)

load("//tensorflow/tensorboard/defs:vulcanize.bzl", "tensorboard_html_binary")
load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

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

tensorboard_html_binary(
    name = "devserver",
    compilation_level = "WHITESPACE_ONLY",
    input_path = "/vz-sorting/test/tests.html",
    output_path = "/vz-sorting/test/tests.html",
    deps = [":test"],
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
