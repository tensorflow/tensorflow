licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//tensorflow:internal"])

load("//tensorflow:tensorflow.bzl", "gpu_py_test")

# Model
py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "ops",
    srcs = ["ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "sagan",
    srcs = ["sagan.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
    ],
)

# Tests
gpu_py_test(
    name = "ops_test",
    size = "small",
    srcs = ["ops_test.py"],
    additional_deps = [
        ":ops",
        "//tensorflow:tensorflow_py",
    ],
)

gpu_py_test(
    name = "sagan_test",
    size = "large",
    srcs = ["sagan_test.py"],
    additional_deps = [
        ":config",
        ":sagan",
        "//tensorflow:tensorflow_py",
    ],
    tags = [
        "optonly",
    ],
)
