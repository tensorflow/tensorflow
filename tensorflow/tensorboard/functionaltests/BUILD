load("@io_bazel_rules_webtesting//web:web.bzl", "web_test_suite")

py_test(
    name = "basic_test_wrapped",
    srcs = ["basic_test.py"],
    data = [
        "//tensorflow/tensorboard",
        "//tensorflow/tensorboard/functionaltests/testdata",
    ],
    main = "basic_test.py",
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = ["@io_bazel_rules_webtesting//testing/web"],
)

web_test_suite(
    name = "basic_test",
    browsers = ["//tensorflow/tensorboard/functionaltests/browsers:headless-chrome"],
    test = ":basic_test_wrapped",
)
