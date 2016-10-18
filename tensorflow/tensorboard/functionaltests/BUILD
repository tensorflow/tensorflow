load("@io_bazel_rules_web//web:web.bzl", "web_test_suite")

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
    deps = ["@io_bazel_rules_web//testing/web"],
)

web_test_suite(
    name = "basic_test",
    browsers = ["@io_bazel_rules_web//browsers:chrome-native"],
    local = True,
    test = ":basic_test_wrapped",
)
