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
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "if_not_windows")

py_library(
    name = "debug_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":debug_data",
        ":debug_utils",
        ":hooks",
        ":local_cli_wrapper",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "debug_pip",
    deps = [
        ":debug_py",
        ":offline_analyzer",
        ":session_debug_testlib",
    ] + if_not_windows([
        ":debug_examples",
    ]),
)

py_library(
    name = "debug_data",
    srcs = ["lib/debug_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "debug_utils",
    srcs = ["lib/debug_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "source_utils",
    srcs = ["lib/source_utils.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "stepper",
    srcs = ["lib/stepper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_utils",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:session_ops",
        "@six_archive//:six",
    ],
)

py_library(
    name = "framework",
    srcs = ["wrappers/framework.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_utils",
        ":stepper",
        "//tensorflow/python:client",
        "//tensorflow/python:errors",
    ],
)

py_library(
    name = "debugger_cli_common",
    srcs = ["cli/debugger_cli_common.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:platform",
        "@six_archive//:six",
    ],
)

py_library(
    name = "command_parser",
    srcs = ["cli/command_parser.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "tensor_format",
    srcs = ["cli/tensor_format.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debugger_cli_common",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cli_shared",
    srcs = ["cli/cli_shared.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":command_parser",
        ":debugger_cli_common",
        ":tensor_format",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "analyzer_cli",
    srcs = ["cli/analyzer_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":command_parser",
        ":debug_data",
        ":debugger_cli_common",
        ":source_utils",
        ":ui_factory",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "profile_analyzer_cli",
    srcs = ["cli/profile_analyzer_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":command_parser",
        ":debug_data",
        ":debugger_cli_common",
        ":source_utils",
        ":ui_factory",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "stepper_cli",
    srcs = ["cli/stepper_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":command_parser",
        ":debugger_cli_common",
        ":stepper",
        ":tensor_format",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "base_ui",
    srcs = ["cli/base_ui.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":command_parser",
        ":debugger_cli_common",
    ],
)

py_library(
    name = "curses_widgets",
    srcs = ["cli/curses_widgets.py"],
    srcs_version = "PY2AND3",
    deps = [":debugger_cli_common"],
)

py_library(
    name = "curses_ui",
    srcs = ["cli/curses_ui.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_ui",
        ":cli_shared",
        ":command_parser",
        ":curses_widgets",
        ":debugger_cli_common",
        ":tensor_format",
        "@six_archive//:six",
    ],
)

py_library(
    name = "readline_ui",
    srcs = ["cli/readline_ui.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_ui",
        ":debugger_cli_common",
    ],
)

py_library(
    name = "ui_factory",
    srcs = ["cli/ui_factory.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":curses_ui",
        ":readline_ui",
    ],
)

py_library(
    name = "dumping_wrapper",
    srcs = ["wrappers/dumping_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":framework",
    ],
)

py_library(
    name = "grpc_wrapper",
    srcs = ["wrappers/grpc_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [":framework"],
)

py_library(
    name = "local_cli_wrapper",
    srcs = ["wrappers/local_cli_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":analyzer_cli",
        ":cli_shared",
        ":debug_data",
        ":debugger_cli_common",
        ":framework",
        ":profile_analyzer_cli",
        ":stepper_cli",
        ":ui_factory",
    ],
)

py_library(
    name = "hooks",
    srcs = ["wrappers/hooks.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_utils",
        ":dumping_wrapper",
        ":framework",
        ":grpc_wrapper",
        ":local_cli_wrapper",
        ":stepper",
        "//tensorflow/python:training",
    ],
)

py_binary(
    name = "offline_analyzer",
    srcs = ["cli/offline_analyzer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":analyzer_cli",
        ":debug_data",
        "//tensorflow/python",  # TODO(b/34059704): remove when fixed
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "debug_examples",
    deps = [
        ":debug_errors",
        ":debug_fibonacci",
        ":debug_mnist",
        ":debug_tflearn_iris",
    ],
)

py_binary(
    name = "debug_fibonacci",
    srcs = ["examples/debug_fibonacci.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "debug_errors",
    srcs = ["examples/debug_errors.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_mnist",
    srcs = ["examples/debug_mnist.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/examples/tutorials/mnist:input_data",
    ],
)

py_binary(
    name = "debug_tflearn_iris",
    srcs = ["examples/debug_tflearn_iris.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "debug_data_test",
    size = "small",
    srcs = [
        "lib/debug_data_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "debug_utils_test",
    size = "small",
    srcs = [
        "lib/debug_utils_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "source_utils_test",
    size = "small",
    srcs = [
        "lib/source_utils_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_utils",
        ":source_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "stepper_test",
    size = "small",
    srcs = [
        "lib/stepper_test.py",
    ],
    additional_deps = [
        ":stepper",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "framework_test",
    size = "small",
    srcs = ["wrappers/framework_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":framework",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "curses_widgets_test",
    size = "small",
    srcs = ["cli/curses_widgets_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":curses_widgets",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "curses_ui_test",
    size = "small",
    srcs = [
        "cli/curses_ui_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
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
    srcs = [
        "cli/readline_ui_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debugger_cli_common",
        ":readline_ui",
        ":ui_factory",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "session_debug_testlib",
    srcs = ["lib/session_debug_testlib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:rnn",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "session_debug_file_test",
    size = "small",
    srcs = ["lib/session_debug_file_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        ":session_debug_testlib",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    tags = ["notsan"],
)

py_test(
    name = "debugger_cli_common_test",
    size = "small",
    srcs = [
        "cli/debugger_cli_common_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "command_parser_test",
    size = "small",
    srcs = [
        "cli/command_parser_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":command_parser",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "tensor_format_test",
    size = "small",
    srcs = [
        "cli/tensor_format_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":tensor_format",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "cli_shared_test",
    size = "small",
    srcs = [
        "cli/cli_shared_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":debugger_cli_common",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "analyzer_cli_test",
    size = "small",
    srcs = [
        "cli/analyzer_cli_test.py",
    ],
    additional_deps = [
        ":analyzer_cli",
        ":command_parser",
        ":debug_data",
        ":debug_utils",
        ":debugger_cli_common",
        ":source_utils",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "profile_analyzer_cli_test",
    size = "small",
    srcs = [
        "cli/profile_analyzer_cli_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":command_parser",
        ":profile_analyzer_cli",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "stepper_cli_test",
    size = "small",
    srcs = [
        "cli/stepper_cli_test.py",
    ],
    additional_deps = [
        ":stepper",
        ":stepper_cli",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "dumping_wrapper_test",
    size = "small",
    srcs = [
        "wrappers/dumping_wrapper_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":dumping_wrapper",
        ":hooks",
        ":stepper",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "local_cli_wrapper_test",
    size = "small",
    srcs = [
        "wrappers/local_cli_wrapper_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":debugger_cli_common",
        ":local_cli_wrapper",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

sh_test(
    name = "examples_test",
    size = "small",
    srcs = ["examples/examples_test.sh"],
    data = [
        ":debug_errors",
        ":debug_fibonacci",
        ":debug_mnist",
        ":debug_tflearn_iris",
        ":offline_analyzer",
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
