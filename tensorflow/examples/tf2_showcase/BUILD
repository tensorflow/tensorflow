package(
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],  # Apache 2.0
)

test_suite(
    name = "all_tests",
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    tests = [
        ":mnist",
    ],
)

py_test(
    name = "mnist",
    srcs = ["mnist.py"],
    python_version = "PY2",
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
