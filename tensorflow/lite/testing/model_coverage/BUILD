package(default_visibility = [
    "//tensorflow/lite:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "model_coverage_lib",
    srcs = ["model_coverage_lib.py"],
    srcs_version = "PY2AND3",
    tags = ["no_pip"],
    deps = [
        "//tensorflow/lite/python:lite",
        "//tensorflow/python:platform",
    ],
)

py_test(
    name = "model_coverage_lib_test",
    srcs = ["model_coverage_lib_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "no_oss",
        "no_pip",
        "no_windows",
        "notap",
    ],
    deps = [
        ":model_coverage_lib",
        "//tensorflow/python:client_testlib",
    ],
)
