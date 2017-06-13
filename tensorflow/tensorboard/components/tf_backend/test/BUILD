package(
    default_testonly = True,
    default_visibility = ["//tensorflow/tensorboard:internal"],
)

load("//tensorflow/tensorboard/defs:web.bzl", "ts_web_library")

licenses(["notice"])  # Apache 2.0

ts_web_library(
    name = "test",
    srcs = [
        "tests.html",
        "backendTests.ts",
        "behaviorTests.ts",
        "requestManagerTests.ts",
    ] + glob(["data/**"]),
    path = "/tf-backend/test",
    deps = [
        "//tensorflow/tensorboard/components/tf_backend",
        "//tensorflow/tensorboard/components/tf_imports:polymer",
        "//tensorflow/tensorboard/components/tf_imports:web_component_tester",
        "//tensorflow/tensorboard/components/tf_imports:webcomponentsjs",
    ],
)

filegroup(
    name = "all_files",
    testonly = 0,
    srcs = glob(["**"]),
    tags = ["notsan"],
)
