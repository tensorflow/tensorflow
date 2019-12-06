# Description:
#   Python Client Code of the TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug_py": Public Python methods and classes of tfdbg.
#   For API documentation, see https://www.tensorflow.org/api_docs/python/tfdbg
#   For a user interface walkthrough, see https://www.tensorflow.org/guide/debugger
# ":grpc_debug_server": Server interface for grpc:// debug URLs.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "if_not_v2", "if_not_windows", "py_binary", "py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["LICENSE"])

py_library(
    name = "debug_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":check_numerics_callback",
        ":debug_data",
        ":debug_events_reader",
        ":debug_events_writer",
        ":debug_gradients",
        ":debug_graphs",
        ":debug_utils",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        ":grpc_debug_server",
        ":grpc_debug_test_server",
        ":hooks",
        ":local_cli_wrapper",
        "//tensorflow/python:debug_ops_gen",
        "//tensorflow/python:util",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_library(
    name = "debug_pip",
    data = [":grpc_tensorflow_server"],
    deps = [
        ":cli_test_utils",
        ":debug_py",
        ":grpc_debug_test_server",
        ":grpc_tensorflow_server_lib",
        ":offline_analyzer_lib",
        ":session_debug_testlib",
        ":source_remote",
    ] + if_not_windows([
        ":debug_examples_v1",
        ":debug_examples_v2",
    ]),
)

py_library(
    name = "op_callbacks_common",
    srcs = ["lib/op_callbacks_common.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "check_numerics_callback",
    srcs = ["lib/check_numerics_callback.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":op_callbacks_common",
        ":source_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:debug_ops_gen",
        "//tensorflow/python:op_callbacks",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dumping_callback",
    srcs = ["lib/dumping_callback.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_events_writer",
        ":op_callbacks_common",
        ":source_utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:debug_ops_gen",
        "//tensorflow/python:op_callbacks",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "dumping_callback_test_lib",
    srcs = ["lib/dumping_callback_test_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":check_numerics_callback",
        ":debug_events_reader",
        ":dumping_callback",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "common",
    srcs = ["lib/common.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "debug_events_reader",
    srcs = ["lib/debug_events_reader.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "debug_events_writer",
    srcs = ["lib/debug_events_writer.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
    ],
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
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
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

py_binary(
    name = "grpc_tensorflow_server",
    srcs = ["lib/grpc_tensorflow_server.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":grpc_tensorflow_server_lib"],
)

py_library(
    name = "grpc_tensorflow_server_lib",
    srcs = [
        "lib/grpc_tensorflow_server.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
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
    name = "framework",
    srcs = ["wrappers/framework.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_utils",
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
        "//third_party/py/numpy",
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
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
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
        ":tensor_format",
        ":ui_factory",
    ],
)

py_library(
    name = "hooks",
    srcs = ["wrappers/hooks.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//tensorflow:internal",
        "//third_party/py/tf_slim:__subpackages__",
    ],
    deps = [
        ":debug_utils",
        ":dumping_wrapper",
        ":framework",
        ":grpc_wrapper",
        ":local_cli_wrapper",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:training",
    ],
)

py_binary(
    name = "offline_analyzer",
    srcs = ["cli/offline_analyzer.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":offline_analyzer_lib"],
)

py_library(
    name = "offline_analyzer_lib",
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
    name = "debug_examples_v1",
    deps = [
        ":debug_errors_lib",
        ":debug_fibonacci_lib",
        ":debug_keras_lib",
    ] + if_not_v2([
        ":debug_mnist_lib",
        ":debug_tflearn_iris_lib",
    ]),
)

py_library(
    name = "debug_examples_v2",
    deps = [
        ":debug_fibonacci_lib",
        ":debug_mnist_lib",
    ],
)

py_binary(
    name = "debug_fibonacci",
    srcs = ["examples/v1/debug_fibonacci.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_fibonacci_lib"],
)

py_binary(
    name = "debug_fibonacci_v2",
    srcs = ["examples/v2/debug_fibonacci_v2.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_fibonacci_lib"],
)

py_library(
    name = "debug_fibonacci_lib",
    srcs = [
        "examples/v1/debug_fibonacci.py",
        "examples/v2/debug_fibonacci_v2.py",
    ],
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
    srcs = ["examples/v1/debug_errors.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_errors_lib"],
)

py_library(
    name = "debug_errors_lib",
    srcs = ["examples/v1/debug_errors.py"],
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_mnist_lib"],
)

py_binary(
    name = "debug_mnist_v1",
    srcs = ["examples/v1/debug_mnist_v1.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_mnist_lib"],
)

py_binary(
    name = "debug_mnist_v2",
    srcs = ["examples/v2/debug_mnist_v2.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_mnist_lib"],
)

py_library(
    name = "debug_mnist_lib",
    srcs = [
        "examples/debug_mnist.py",
        "examples/v1/debug_mnist_v1.py",
        "examples/v2/debug_mnist_v2.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "debug_tflearn_iris",
    srcs = ["examples/v1/debug_tflearn_iris.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_tflearn_iris_lib"],
)

py_library(
    name = "debug_tflearn_iris_lib",
    srcs = ["examples/v1/debug_tflearn_iris.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

py_binary(
    name = "debug_keras",
    srcs = ["examples/v1/debug_keras.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [":debug_keras_lib"],
)

py_library(
    name = "debug_keras_lib",
    srcs = ["examples/v1/debug_keras.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":debug_py",
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "common_test",
    size = "small",
    srcs = ["lib/common_test.py"],
    python_version = "PY3",
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
    name = "debug_events_writer_test",
    size = "medium",
    srcs = ["lib/debug_events_writer_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",  # b/142475891
    ],
    deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback_test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "debug_graphs_test",
    size = "small",
    srcs = ["lib/debug_graphs_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
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
    name = "check_numerics_callback_test",
    size = "medium",
    srcs = ["lib/check_numerics_callback_test.py"],
    additional_deps = [
        ":check_numerics_callback",
        "//third_party/py/numpy",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    python_version = "PY3",
)

cuda_py_test(
    name = "distributed_callbacks_test",
    size = "medium",
    srcs = ["lib/distributed_callbacks_test.py"],
    additional_deps = [
        ":check_numerics_callback",
        ":debug_events_writer",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        "//third_party/py/numpy",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/keras",
    ],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "guitar",
        "multi_and_single_gpu",
        "no_rocm",
        "no_windows",  # TODO(b/142475891): Enable this test on Windows.
        "no_windows_gpu",  # TODO(b/130551176)
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

cuda_py_test(
    name = "dumping_callback_test",
    size = "medium",
    srcs = ["lib/dumping_callback_test.py"],
    additional_deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        "//third_party/py/numpy",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras",
    ],
    python_version = "PY3",
    shard_count = 8,
    tags = [
        "no_windows",  # TODO(b/142475891): Enable this test on Windows.
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

cuda_py_test(
    name = "debug_v2_ops_test",
    size = "medium",
    srcs = ["lib/debug_v2_ops_test.py"],
    additional_deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback_test_lib",
        "//third_party/py/numpy",
        "//tensorflow/python:debug_ops_gen",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras",
    ],
    python_version = "PY3",
    tags = [
        "no_windows",  # b/142475891
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
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

py_test(
    name = "debug_utils_test",
    size = "small",
    srcs = ["lib/debug_utils_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":debug_utils",
        ":source_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:cond_v2",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python:while_v2",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "source_remote_test",
    size = "small",
    srcs = ["lib/source_remote_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
        "oss_serial",
    ],
    deps = [
        ":grpc_debug_test_server",
        ":source_remote",
        ":source_utils",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "framework_test",
    size = "medium",
    srcs = ["wrappers/framework_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
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
    name = "debug_grappler_test",
    size = "small",
    srcs = ["lib/debug_grappler_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,  # Tests TF:Classic implementation.
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
    python_version = "PY3",
    tags = ["notsan"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
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
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
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
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

py_test(
    name = "debugger_cli_common_test",
    size = "small",
    srcs = ["cli/debugger_cli_common_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":debugger_cli_common",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "cli_config_test",
    size = "small",
    srcs = ["cli/cli_config_test.py"],
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
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
    python_version = "PY3",
    tags = ["no_windows"],  # TODO: needs investigation on Windows
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

py_test(
    name = "profile_analyzer_cli_test",
    size = "small",
    srcs = ["cli/profile_analyzer_cli_test.py"],
    python_version = "PY3",
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
    name = "session_debug_grpc_test",
    size = "medium",
    srcs = ["lib/session_debug_grpc_test.py"],
    additional_deps = [
        ":debug_data",
        ":debug_utils",
        ":dumping_wrapper",
        ":grpc_debug_test_server",
        ":grpc_wrapper",
        ":hooks",
        ":session_debug_testlib",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    python_version = "PY3",
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "no_windows",
        "notsan",
        "oss_serial",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

cuda_py_test(
    name = "grpc_large_data_test",
    size = "medium",
    srcs = ["lib/grpc_large_data_test.py"],
    additional_deps = [
        ":dumping_wrapper",
        ":grpc_debug_test_server",
        ":grpc_wrapper",
        ":session_debug_testlib",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
    python_version = "PY3",
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "no_windows",
        "noasan",  # Times out due to size of test (b/73731462).
        "optonly",  # Test flaky (b/80130873)
        "oss_serial",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
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
    grpc_enabled = True,
    python_version = "PY3",
    tags = [
        "no_oss",  # Incompatible with bazel_pip.
        "no_windows",
        "notsan",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
)

py_test(
    name = "dumping_wrapper_test",
    size = "small",
    srcs = ["wrappers/dumping_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":debug_data",
        ":dumping_wrapper",
        ":framework",
        ":hooks",
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
    python_version = "PY3",
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
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "disk_usage_test",
    size = "small",
    srcs = ["wrappers/disk_usage_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":dumping_wrapper",
        ":hooks",
        "//tensorflow/python:client",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

sh_test(
    name = "examples_v1_test",
    size = "medium",
    srcs = ["examples/v1/examples_v1_test.sh"],
    data = [
        ":debug_errors",
        ":debug_fibonacci",
        ":debug_keras",
        ":debug_mnist",
        ":debug_tflearn_iris",
        ":offline_analyzer",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/143150907)
        "nomsan",  # TODO(b/143150907)
        "v1only",
    ],
)

sh_test(
    name = "examples_v2_test",
    size = "medium",
    srcs = ["examples/v2/examples_v2_test.sh"],
    data = [
        ":debug_fibonacci_v2",
        ":debug_mnist_v2",
    ],
    tags = [
        "no_windows",
    ],
)
