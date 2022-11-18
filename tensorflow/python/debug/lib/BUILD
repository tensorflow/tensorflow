load("//tensorflow:tensorflow.default.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "py_binary", "py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "op_callbacks_common",
    srcs = ["op_callbacks_common.py"],
    srcs_version = "PY3",
)

py_library(
    name = "check_numerics_callback",
    srcs = ["check_numerics_callback.py"],
    srcs_version = "PY3",
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
    srcs = ["dumping_callback.py"],
    srcs_version = "PY3",
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
    srcs = ["dumping_callback_test_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":check_numerics_callback",
        ":debug_events_reader",
        ":dumping_callback",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
)

py_library(
    name = "debug_events_reader",
    srcs = ["debug_events_reader.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:lib",
    ],
)

py_library(
    name = "debug_events_monitors",
    srcs = ["debug_events_monitors.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:lib",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "debug_events_writer",
    srcs = ["debug_events_writer.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
    ],
)

py_library(
    name = "debug_graphs",
    srcs = ["debug_graphs.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework",
        "//tensorflow/python:op_def_registry",
        "//tensorflow/python:platform",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "debug_data",
    srcs = ["debug_data.py"],
    srcs_version = "PY3",
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
    ],
)

py_library(
    name = "debug_gradients",
    srcs = ["debug_gradients.py"],
    srcs_version = "PY3",
    deps = [
        ":debug_data",
        ":debug_graphs",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

py_library(
    name = "debug_utils",
    srcs = ["debug_utils.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "grpc_tensorflow_server",
    srcs = ["grpc_tensorflow_server.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":grpc_tensorflow_server_lib"],
)

py_library(
    name = "grpc_tensorflow_server_lib",
    srcs = [
        "grpc_tensorflow_server.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "source_utils",
    srcs = ["source_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":profiling",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "source_remote",
    srcs = ["source_remote.py"],
    srcs_version = "PY3",
    deps = [
        ":common",
        ":debug_service_pb2_grpc",
        "//tensorflow/core/debug:debug_service_proto_py",
        "//tensorflow/python/profiler:tfprof_logger",
    ],
)

py_library(
    name = "profiling",
    srcs = ["profiling.py"],
    srcs_version = "PY3",
)

py_test(
    name = "common_test",
    size = "small",
    srcs = ["common_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":common",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "debug_events_monitors_test",
    size = "medium",
    srcs = ["debug_events_monitors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",  # b/142475891
    ],
    deps = [
        ":debug_events_monitors",
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "debug_events_writer_test",
    size = "medium",
    srcs = ["debug_events_writer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs = ["debug_graphs_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":debug_graphs",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "debug_data_test",
    size = "small",
    srcs = ["debug_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_windows"],  # TODO(b/184424727): Enable this test on Windows.
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
    srcs = ["check_numerics_callback_test.py"],
    python_version = "PY3",
    tags = [
        "no_mac",  # TODO(b/175322370): Detected Infinity or NaN in output 0 of graph op "RealDiv"
        "no_windows",
    ],
    deps = [
        ":check_numerics_callback",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "dumping_callback_test",
    size = "medium",
    srcs = ["dumping_callback_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/142475891): Enable this test on Windows.
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback",
        ":dumping_callback_test_lib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "debug_v2_ops_test",
    size = "medium",
    srcs = ["debug_v2_ops_test.py"],
    python_version = "PY3",
    tags = ["no_windows_gpu"],
    deps = [
        ":debug_events_reader",
        ":debug_events_writer",
        ":dumping_callback_test_lib",
        "//tensorflow/python:debug_ops_gen",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "debug_gradients_test",
    size = "small",
    srcs = ["debug_gradients_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
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
    srcs = ["debug_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    srcs = ["source_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
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
    srcs = ["source_remote_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
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
    name = "profiling_test",
    size = "small",
    srcs = ["profiling_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":profiling",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "session_debug_testlib",
    srcs = ["session_debug_testlib.py"],
    srcs_version = "PY3",
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
    ],
)

py_library(
    name = "debug_service_pb2_grpc",
    srcs = ["debug_service_pb2_grpc.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/debug:debug_service_proto_py",
    ],
)

py_library(
    name = "grpc_debug_server",
    srcs = ["grpc_debug_server.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":debug_graphs",
        ":debug_service_pb2_grpc",
        "//tensorflow/core/debug:debug_service_proto_py",
    ],
)

py_library(
    name = "grpc_debug_test_server",
    srcs = ["grpc_debug_test_server.py"],
    srcs_version = "PY3",
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
    srcs = ["debug_grappler_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,  # Tests TF:Classic implementation.
    deps = [
        ":debug_data",
        ":debug_utils",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "session_debug_file_test",
    size = "small",
    srcs = ["session_debug_file_test.py"],
    python_version = "PY3",
    tags = ["notsan"],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
        ":debug_data",
        ":debug_utils",
        ":session_debug_testlib",
        "//tensorflow/python:client",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "debug_graph_reconstruction_test",
    size = "small",
    srcs = ["debug_graph_reconstruction_test.py"],
    python_version = "PY3",
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
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
    srcs = ["session_debug_multi_gpu_test.py"],
    python_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
        "no_windows_gpu",
    ],
    xla_enable_strict_auto_jit = False,  # Node names are different with autojit
    deps = [
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
