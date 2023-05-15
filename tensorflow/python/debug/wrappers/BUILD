load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "framework",
    srcs = ["framework.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:errors",
        "//tensorflow/python:training",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_utils",
        "//tensorflow/python/framework:stack",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "dumping_wrapper",
    srcs = ["dumping_wrapper.py"],
    srcs_version = "PY3",
    visibility = ["//tensorflow:internal"],
    deps = [
        ":framework",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_library(
    name = "grpc_wrapper",
    srcs = ["grpc_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        ":framework",
        "//tensorflow/python/debug/lib:common",
        "//tensorflow/python/debug/lib:source_remote",
    ],
)

py_strict_library(
    name = "local_cli_wrapper",
    srcs = ["local_cli_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        ":framework",
        "//tensorflow/python/debug/cli:analyzer_cli",
        "//tensorflow/python/debug/cli:cli_config",
        "//tensorflow/python/debug/cli:cli_shared",
        "//tensorflow/python/debug/cli:command_parser",
        "//tensorflow/python/debug/cli:debugger_cli_common",
        "//tensorflow/python/debug/cli:profile_analyzer_cli",
        "//tensorflow/python/debug/cli:ui_factory",
        "//tensorflow/python/debug/lib:common",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/lib/io:lib",
    ],
)

py_strict_library(
    name = "hooks",
    srcs = ["hooks.py"],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":dumping_wrapper",
        ":framework",
        ":grpc_wrapper",
        ":local_cli_wrapper",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/debug/lib:debug_utils",
        "//tensorflow/python/training:session_run_hook",
    ],
)

py_strict_test(
    name = "framework_test",
    size = "medium",
    srcs = ["framework_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [],
    deps = [
        ":framework",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/util:tf_decorator",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "dumping_wrapper_test",
    size = "small",
    srcs = ["dumping_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dumping_wrapper",
        ":framework",
        ":hooks",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_test(
    name = "local_cli_wrapper_test",
    size = "small",
    srcs = ["local_cli_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":local_cli_wrapper",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_v1",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/cli:cli_config",
        "//tensorflow/python/debug/cli:cli_shared",
        "//tensorflow/python/debug/cli:debugger_cli_common",
        "//tensorflow/python/debug/cli:ui_factory",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/lib/io:lib",
        "//tensorflow/python/training:session_run_hook",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "disk_usage_test",
    size = "small",
    srcs = ["disk_usage_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dumping_wrapper",
        ":hooks",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
    ],
)
