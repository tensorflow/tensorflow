load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "framework",
    srcs = ["framework.py"],
    strict_deps = True,
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_utils",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:stack",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:nest",
    ],
)

py_library(
    name = "dumping_wrapper",
    srcs = ["dumping_wrapper.py"],
    strict_deps = True,
    visibility = ["//tensorflow:internal"],
    deps = [
        ":framework",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/platform:gfile",
    ],
)

py_library(
    name = "grpc_wrapper",
    srcs = ["grpc_wrapper.py"],
    strict_deps = True,
    deps = [
        ":framework",
        "//tensorflow/python/debug/lib:common",
        "//tensorflow/python/debug/lib:source_remote",
    ],
)

py_library(
    name = "local_cli_wrapper",
    srcs = ["local_cli_wrapper.py"],
    strict_deps = True,
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
        "//tensorflow/python/lib/io:file_io",
    ],
)

py_library(
    name = "hooks",
    srcs = ["hooks.py"],
    strict_deps = True,
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

py_test(
    name = "framework_test",
    size = "medium",
    srcs = ["framework_test.py"],
    strict_deps = True,
    tags = [
        "no_rocm",
    ],
    deps = [
        ":framework",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_test(
    name = "dumping_wrapper_test",
    size = "small",
    srcs = ["dumping_wrapper_test.py"],
    strict_deps = True,
    deps = [
        ":dumping_wrapper",
        ":framework",
        ":hooks",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/platform:gfile",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/training:monitored_session",
    ],
)

py_test(
    name = "local_cli_wrapper_test",
    size = "small",
    srcs = ["local_cli_wrapper_test.py"],
    strict_deps = True,
    deps = [
        ":local_cli_wrapper",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/client:session",
        "//tensorflow/python/debug/cli:cli_config",
        "//tensorflow/python/debug/cli:cli_shared",
        "//tensorflow/python/debug/cli:debugger_cli_common",
        "//tensorflow/python/debug/cli:ui_factory",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variable_v1",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/training:monitored_session",
        "//tensorflow/python/training:session_run_hook",
    ],
)

py_test(
    name = "disk_usage_test",
    size = "small",
    srcs = ["disk_usage_test.py"],
    strict_deps = True,
    deps = [
        ":dumping_wrapper",
        ":hooks",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow/python/client:session",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/training:monitored_session",
    ],
)
