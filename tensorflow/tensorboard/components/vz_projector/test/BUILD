package(
    default_testonly = True,
    default_visibility = ["//tensorflow/tensorboard:internal"],
)

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "test",
    srcs = [
        "assert.ts",
        "data-provider_test.ts",
        "data_test.ts",
        "sptree_test.ts",
        "tests.html",
        "util_test.ts",
        # "scatterPlotRectangleSelector_test.ts",
        # "vz-projector-projections-panel_test.ts",
    ],
    path = "/vz-projector/test",
    deps = [
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/tf_imports:web_component_tester",
        "//tensorflow/tensorboard/components/tf_imports:webcomponentsjs",
        "//tensorflow/tensorboard/components/vz_projector",
    ],
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
