load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "profile_analyzer_cli",
    srcs = ["profile_analyzer_cli.py"],
    srcs_version = "PY3",
    deps = [
        ":cli_shared",
        ":command_parser",
        ":debugger_cli_common",
        ":ui_factory",
        "//tensorflow/python/debug/lib:profiling",
        "//tensorflow/python/debug/lib:source_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "base_ui",
    srcs = ["base_ui.py"],
    srcs_version = "PY3",
    deps = [
        ":cli_config",
        ":command_parser",
        ":debugger_cli_common",
    ],
)

py_library(
    name = "curses_widgets",
    srcs = ["curses_widgets.py"],
    srcs_version = "PY3",
    deps = [":debugger_cli_common"],
)

py_library(
    name = "curses_ui",
    srcs = ["curses_ui.py"],
    srcs_version = "PY3",
    deps = [
        ":base_ui",
        ":cli_shared",
        ":command_parser",
        ":curses_widgets",
        ":debugger_cli_common",
        ":tensor_format",
    ],
)

py_library(
    name = "readline_ui",
    srcs = ["readline_ui.py"],
    srcs_version = "PY3",
    deps = [
        ":base_ui",
        ":debugger_cli_common",
    ],
)

py_library(
    name = "ui_factory",
    srcs = ["ui_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":curses_ui",
        ":readline_ui",
    ],
)

py_library(
    name = "command_parser",
    srcs = ["command_parser.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tensor_format",
    srcs = ["tensor_format.py"],
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python/debug/lib:debug_data",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cli_shared",
    srcs = ["cli_shared.py"],
    srcs_version = "PY3",
    deps = [
        ":command_parser",
        ":debugger_cli_common",
        ":tensor_format",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//tensorflow/python/debug/lib:common",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/debug/lib:debug_data",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "analyzer_cli",
    srcs = ["analyzer_cli.py"],
    srcs_version = "PY3",
    deps = [
        ":cli_config",
        ":cli_shared",
        ":command_parser",
        ":debugger_cli_common",
        ":evaluator",
        ":ui_factory",
        "//tensorflow/python/debug/lib:debug_graphs",
        "//tensorflow/python/debug/lib:source_utils",
    ],
)

py_library(
    name = "cli_config",
    srcs = ["cli_config.py"],
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python/platform",
    ],
)

py_library(
    name = "debugger_cli_common",
    srcs = ["debugger_cli_common.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:platform",
        "//tensorflow/python:pywrap_tf_session",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "offline_analyzer",
    srcs = ["offline_analyzer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":offline_analyzer_lib"],
)

py_library(
    name = "offline_analyzer_lib",
    srcs = ["offline_analyzer.py"],
    srcs_version = "PY3",
    deps = [
        ":analyzer_cli",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
        "//tensorflow/python/debug/lib:debug_data",
    ],
)

py_test(
    name = "curses_ui_test",
    size = "small",
    srcs = ["curses_ui_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":cli_config",
        ":cli_test_utils",
        ":curses_ui",
        ":debugger_cli_common",
        ":tensor_format",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "readline_ui_test",
    size = "small",
    srcs = ["readline_ui_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_windows"],  # TODO(b/214427155)
    deps = [
        ":cli_config",
        ":debugger_cli_common",
        ":readline_ui",
        ":ui_factory",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "debugger_cli_common_test",
    size = "small",
    srcs = ["debugger_cli_common_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:pywrap_tf_session",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "cli_config_test",
    size = "small",
    srcs = ["cli_config_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_config",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/lib/io:lib",
    ],
)

py_test(
    name = "command_parser_test",
    size = "small",
    srcs = ["command_parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":command_parser",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "tensor_format_test",
    size = "small",
    srcs = ["tensor_format_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_test_utils",
        ":tensor_format",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/debug/lib:debug_data",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "cli_shared_test",
    size = "small",
    srcs = ["cli_shared_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_shared",
        ":debugger_cli_common",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
        "//tensorflow/python/framework:constant_op",
    ],
)

py_test(
    name = "evaluator_test",
    size = "small",
    srcs = [
        "evaluator_test.py",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_windows"],  # TODO(b/184424727): Enable this test on Windows.
    deps = [
        ":evaluator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/debug/lib:debug_data",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cli_test_utils",
    srcs = ["cli_test_utils.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/numpy"],
)

cuda_py_test(
    name = "analyzer_cli_test",
    size = "small",
    srcs = ["analyzer_cli_test.py"],
    python_version = "PY3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":analyzer_cli",
        ":cli_config",
        ":cli_shared",
        ":cli_test_utils",
        ":command_parser",
        ":debugger_cli_common",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/debug/lib:debug_utils",
        "//tensorflow/python/debug/lib:source_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/lib/io:lib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "profile_analyzer_cli_test",
    size = "small",
    srcs = ["profile_analyzer_cli_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        ":profile_analyzer_cli",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/framework:constant_op",
    ],
)
