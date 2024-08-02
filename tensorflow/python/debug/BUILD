# Description:
#   Python Client Code of the TensorFlow Debugger (tfdbg).
#
# Public target(s):
#
# ":debug_py": Public Python methods and classes of tfdbg.
#   For API documentation, see https://www.tensorflow.org/api_docs/python/tfdbg
#   For a user interface walkthrough, see https://www.tensorflow.org/guide/debugger
# ":grpc_debug_server": Server interface for grpc:// debug URLs.

load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_library(
    name = "debug_py",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],  # Generated files need dependencies that build_cleaner would remove
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/python/debug/lib:check_numerics_callback",
        "//tensorflow/python/debug/lib:debug_data",
        "//tensorflow/python/debug/lib:debug_events_monitors",
        "//tensorflow/python/debug/lib:debug_events_reader",
        "//tensorflow/python/debug/lib:debug_gradients",
        "//tensorflow/python/debug/lib:debug_graphs",
        "//tensorflow/python/debug/lib:debug_utils",
        "//tensorflow/python/debug/lib:dumping_callback",
        "//tensorflow/python/debug/lib:dumping_callback_test_lib",
        "//tensorflow/python/debug/lib:grpc_debug_server",
        "//tensorflow/python/debug/lib:grpc_debug_test_server",
        "//tensorflow/python/debug/wrappers:dumping_wrapper",
        "//tensorflow/python/debug/wrappers:framework",
        "//tensorflow/python/debug/wrappers:grpc_wrapper",
        "//tensorflow/python/debug/wrappers:hooks",
        "//tensorflow/python/debug/wrappers:local_cli_wrapper",
        "//tensorflow/python/util:all_util",
    ],
)

# Transitive dependencies of this target will be included in the pip package.
py_strict_library(
    name = "debug_pip",
    data = ["//tensorflow/python/debug/lib:grpc_tensorflow_server"],
    srcs_version = "PY3",
    deps = [
        ":debug_py",
        "//tensorflow/python/debug/cli:cli_test_utils",
        "//tensorflow/python/debug/cli:offline_analyzer_lib",
        "//tensorflow/python/debug/lib:grpc_debug_test_server",
        "//tensorflow/python/debug/lib:grpc_tensorflow_server_lib",
        "//tensorflow/python/debug/lib:session_debug_testlib",
        "//tensorflow/python/debug/lib:source_remote",
    ],
)
