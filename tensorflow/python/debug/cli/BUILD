load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
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

py_strict_library(
    name = "base_ui",
    srcs = ["base_ui.py"],
    srcs_version = "PY3",
    deps = [
        ":cli_config",
        ":command_parser",
        ":debugger_cli_common",
    ],
)

py_strict_library(
    name = "readline_ui",
    srcs = ["readline_ui.py"],
    srcs_version = "PY3",
    deps = [
        ":base_ui",
        ":debugger_cli_common",
    ],
)

py_strict_library(
    name = "ui_factory",
    srcs = ["ui_factory.py"],
    srcs_version = "PY3",
    deps = [
        ":readline_ui",  # build_cleaner keep.
    ],
)

py_strict_library(
    name = "command_parser",
    srcs = ["command_parser.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "tensor_format",
    srcs = ["tensor_format.py"],
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python/debug/lib:debug_data",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "cli_shared",
    srcs = ["cli_shared.py"],
    srcs_version = "PY3",
    deps = [
        ":command_parser",
        ":debugger_cli_common",
        ":tensor_format",
        "//tensorflow/python/debug/lib:common",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:gfile",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "evaluator",
    srcs = ["evaluator.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/debug/lib:debug_data",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
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

py_strict_library(
    name = "cli_config",
    srcs = ["cli_config.py"],
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python/platform:gfile",
    ],
)

py_strict_library(
    name = "debugger_cli_common",
    srcs = ["debugger_cli_common.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/platform:gfile",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "offline_analyzer",
    srcs = ["offline_analyzer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":offline_analyzer_lib"],
)

py_strict_library(
    name = "offline_analyzer_lib",
    srcs = ["offline_analyzer.py"],
    srcs_version = "PY3",
    deps = [
        ":analyzer_cli",
        "//tensorflow/python/debug/lib:debug_data",
        "@absl_py//absl:app",
    ],
)

py_strict_test(
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
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "debugger_cli_common_test",
    size = "small",
    srcs = ["debugger_cli_common_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python/client:pywrap_tf_session",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "cli_config_test",
    size = "small",
    srcs = ["cli_config_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_config",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "command_parser_test",
    size = "small",
    srcs = ["command_parser_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":command_parser",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "tensor_format_test",
    size = "small",
    srcs = ["tensor_format_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_test_utils",
        ":tensor_format",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "cli_shared_test",
    size = "small",
    srcs = ["cli_shared_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cli_shared",
        ":debugger_cli_common",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
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
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "cli_test_utils",
    srcs = ["cli_test_utils.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/numpy"],
)

cuda_py_strict_test(
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
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/debug/lib:debug_utils",
        "//tensorflow/python/debug/lib:source_utils",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "profile_analyzer_cli_test",
    size = "small",
    srcs = ["profile_analyzer_cli_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":debugger_cli_common",
        ":profile_analyzer_cli",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:tf_inspect",
    ],
)
