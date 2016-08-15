# Description:
#   Python Client Code of the TensorFlow Debugger (tfdbg).

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "debug_utils",
    srcs = ["debug_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:session",
    ],
)

py_test(
    name = "debug_utils_test",
    size = "small",
    srcs = [
        "debug_utils_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_utils",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:session",
    ],
)

cuda_py_test(
    name = "session_debug_test",
    size = "small",
    srcs = [
        "session_debug_test.py",
    ],
    additional_deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:session",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
