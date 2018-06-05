# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "lms_py",
    srcs = [
        "__init__.py",
        "python/lms.py",
        "python/topos.py",
        "python/lms_session_run_hook.py",

    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "lms_pip",
    deps = [
        ":lms_py",
        "//tensorflow/contrib/lms/examples:lms_examples",
    ],
)

py_test(
    name = "lms_test",
    srcs = ["python/lms_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lms_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/contrib/graph_editor:graph_editor_py"
    ],
)

py_test(
    name = "topos_test",
    srcs = ["python/topos_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lms_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/contrib/graph_editor:graph_editor_py"
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
