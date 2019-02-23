licenses(["notice"])  # Apache 2.0

package(
    default_visibility = ["//visibility:private"],
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
    tags = [
        "manual",
        "no_oss",
        "notap",
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)
