# Description:
#   Contains `SessionRunHook`s for use with `MonitoredSession` and the
#   wrappers around it.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "hooks",
    srcs = [
        "__init__.py",
        "python/hooks/__init__.py",
        "python/hooks/profiler_hook.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "profiler_hook_test",
    size = "small",
    srcs = ["python/hooks/profiler_hook_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":session_run_hooks",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
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
)
