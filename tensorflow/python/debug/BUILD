# Description:
#   Python Client Code of the TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug_py": Public Python methods and classes of tfdbg.
#   For API documentation, see https://www.tensorflow.org/api_docs/python/tfdbg
#   For a user interface walkthrough, see https://www.tensorflow.org/programmers_guide/debugger
# ":grpc_debug_server": Server interface for grpc:// debug URLs.

package(
    default_visibility = ["//tensorflow:internal"],
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
        ":debug_gradients",
        ":debug_graphs",
        ":debug_utils",
        ":grpc_debug_server",
        ":grpc_debug_test_server",
        ":hooks",
        ":local_cli_wrapper",
        "//tensorflow/python:util",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "debug_pip",
    deps = [
        ":cli_test_utils",
        ":debug_py",
        ":grpc_debug_test_server",
        ":offline_analyzer",
        ":session_debug_testlib",
        ":source_remote",
    ] + if_not_windows([
        ":debug_examples",
    ]),
)

py_library(
    name = "common",
    srcs = ["lib/common.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "debug_graphs",
    srcs = ["lib/debug_graphs.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:op_def_registry",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_util",
        "@six_archive//:six",
    ],
)

py_library(
    name = "debug_data",
    srcs = ["lib/debug_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_graphs",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:op_def_registry",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_util",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "debug_gradients",
    srcs = ["lib/debug_gradients.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_graphs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "@six_archive//:six",
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
    deps = [
        ":profiling",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "source_remote",
    srcs = ["lib/source_remote.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        ":debug_service_pb2_grpc",
        "//tensorflow/core/debug:debug_service_proto_py",
        "//tensorflow/python/profiler:tfprof_logger",
    ],
)

py_library(
    name = "stepper",
    srcs = ["lib/stepper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_graphs",
        ":debug_utils",
        "//tensorflow/core:protos_all_py",
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:training",
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
    name = "cli_config",
    srcs = ["cli/cli_config.py"],
    srcs_version = "PY2AND3",
    deps = [":debugger_cli_common"],
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
        ":common",
        ":debugger_cli_common",
        ":tensor_format",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "evaluator",
    srcs = ["cli/evaluator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "analyzer_cli",
    srcs = ["cli/analyzer_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_config",
        ":cli_shared",
        ":command_parser",
        ":debug_graphs",
        ":debugger_cli_common",
        ":evaluator",
        ":source_utils",
        ":ui_factory",
        "@six_archive//:six",
    ],
)

py_library(
    name = "profiling",
    srcs = ["lib/profiling.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "profile_analyzer_cli",
    srcs = ["cli/profile_analyzer_cli.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":command_parser",
        ":debugger_cli_common",
        ":profiling",
        ":source_utils",
        ":ui_factory",
        "//third_party/py/numpy",
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
        ":cli_config",
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "grpc_wrapper",
    srcs = ["wrappers/grpc_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        ":framework",
        ":source_remote",
    ],
)

py_library(
    name = "local_cli_wrapper",
    srcs = ["wrappers/local_cli_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":analyzer_cli",
        ":cli_shared",
        ":command_parser",
        ":common",
        ":debug_data",
        ":debugger_cli_common",
        ":framework",
        ":profile_analyzer_cli",
        ":stepper_cli",
        ":tensor_format",
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
        "//tensorflow/core:protos_all_py",
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
    name = "common_test",
    size = "small",
    srcs = ["lib/common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":common",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "debug_graphs_test",
    size = "small",
    srcs = ["lib/debug_graphs_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_graphs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "debug_data_test",
    size = "small",
    srcs = ["lib/debug_data_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "debug_gradients_test",
    size = "small",
    srcs = ["lib/debug_gradients_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_gradients",
        ":debug_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "debug_utils_test",
    size = "small",
    srcs = ["lib/debug_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "source_utils_test",
    size = "small",
    srcs = ["lib/source_utils_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_utils",
        ":source_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "source_remote_test",
    size = "small",
    srcs = ["lib/source_remote_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "nomac",
        "oss_serial",
    ],
    deps = [
        ":grpc_debug_test_server",
        ":source_remote",
        ":source_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "stepper_test",
    size = "small",
    srcs = ["lib/stepper_test.py"],
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
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "profiling_test",
    size = "small",
    srcs = ["lib/profiling_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":profiling",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "curses_ui_test",
    size = "small",
    srcs = ["cli/curses_ui_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
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
    srcs = ["cli/readline_ui_test.py"],
    srcs_version = "PY2AND3",
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

py_library(
    name = "session_debug_testlib",
    srcs = ["lib/session_debug_testlib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_graphs",
        ":debug_utils",
        "//tensorflow/core:protos_all_py",
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
        "//tensorflow/python:rnn_cell",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "debug_service_pb2_grpc",
    srcs = ["lib/debug_service_pb2_grpc.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core/debug:debug_service_proto_py",
    ],
)

py_library(
    name = "grpc_debug_server",
    srcs = ["lib/grpc_debug_server.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":debug_graphs",
        ":debug_service_pb2_grpc",
        "//tensorflow/core/debug:debug_service_proto_py",
        "@six_archive//:six",
    ],
)

py_library(
    name = "grpc_debug_test_server",
    srcs = ["lib/grpc_debug_test_server.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_utils",
        ":grpc_debug_server",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:variables",
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

cuda_py_test(
    name = "debug_graph_reconstruction_test",
    size = "small",
    srcs = ["lib/debug_graph_reconstruction_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "session_debug_multi_gpu_test",
    size = "small",
    srcs = ["lib/session_debug_multi_gpu_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "debugger_cli_common_test",
    size = "small",
    srcs = ["cli/debugger_cli_common_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "cli_config_test",
    size = "small",
    srcs = ["cli/cli_config_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_config",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "command_parser_test",
    size = "small",
    srcs = ["cli/command_parser_test.py"],
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
    srcs = ["cli/tensor_format_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_test_utils",
        ":debug_data",
        ":tensor_format",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "cli_shared_test",
    size = "small",
    srcs = ["cli/cli_shared_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":debugger_cli_common",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "evaluator_test",
    size = "small",
    srcs = [
        "cli/evaluator_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":evaluator",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "cli_test_utils",
    srcs = ["cli/cli_test_utils.py"],
    srcs_version = "PY2AND3",
)

cuda_py_test(
    name = "analyzer_cli_test",
    size = "small",
    srcs = ["cli/analyzer_cli_test.py"],
    additional_deps = [
        ":analyzer_cli",
        ":cli_config",
        ":cli_test_utils",
        ":command_parser",
        ":debug_data",
        ":debug_utils",
        ":debugger_cli_common",
        ":source_utils",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:array_ops",
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
    srcs = ["cli/profile_analyzer_cli_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debugger_cli_common",
        ":profile_analyzer_cli",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "stepper_cli_test",
    size = "small",
    srcs = ["cli/stepper_cli_test.py"],
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

cuda_py_test(
    name = "session_debug_grpc_test",
    size = "large",
    srcs = ["lib/session_debug_grpc_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        ":dumping_wrapper",
        ":grpc_debug_test_server",
        ":grpc_wrapper",
        ":hooks",
        ":session_debug_testlib",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "no_windows",
        "notsan",
        "oss_serial",
    ],
)

# TODO(cais): Run the test in OSS, perhaps through a sh_test.
cuda_py_test(
    name = "dist_session_debug_grpc_test",
    size = "medium",
    srcs = ["lib/dist_session_debug_grpc_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        ":dumping_wrapper",
        ":grpc_debug_test_server",
        ":grpc_wrapper",
        ":hooks",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    data = ["//tensorflow/tools/dist_test/server:grpc_tensorflow_server"],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Incompatible with bazel_pip.
        "no_windows",
        "nomac",  # TODO(cais): Install of futures and grpcio on all macs.
        "notsan",
    ],
)

py_test(
    name = "dumping_wrapper_test",
    size = "small",
    srcs = ["wrappers/dumping_wrapper_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":dumping_wrapper",
        ":framework",
        ":hooks",
        ":stepper",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "local_cli_wrapper_test",
    size = "small",
    srcs = ["wrappers/local_cli_wrapper_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":cli_shared",
        ":debugger_cli_common",
        ":local_cli_wrapper",
        ":ui_factory",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

sh_test(
    name = "examples_test",
    size = "medium",
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
