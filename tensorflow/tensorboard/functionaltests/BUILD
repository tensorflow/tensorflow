load("@io_bazel_rules_webtesting//web:py.bzl", "py_web_test_suite")

py_web_test_suite(
    name = "basic_test",
    srcs = ["basic_test.py"],
    browsers = ["//tensorflow/tensorboard/functionaltests/browsers:headless-chrome"],
    data = [
        "//tensorflow/tensorboard",
        "//tensorflow/tensorboard/functionaltests/testdata",
    ],
    srcs_version = "PY2AND3",
    deps = ["@io_bazel_rules_webtesting//testing/web"],
)
