load("//tensorflow:tensorflow.bzl", "pytype_strict_library")
load("//tensorflow:tensorflow.bzl", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow/core/function:__subpackages__"],
)

licenses(["notice"])

# TODO(b/220191419): Remove this once the `runtime_client_pybind` build rule is moved to this file.
exports_files(
    ["runtime_client_pybind.cc"],
    visibility = ["//tensorflow:internal"],
)

cc_library(
    name = "runtime_client_cc",
    srcs = [
        "runtime_client.cc",
    ],
    hdrs = [
        "runtime_client.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/c/eager:abstract_tensor_handle",
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/c/eager:immediate_execution_operation",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:device_mgr",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:core",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/importexport:export",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
    # TODO(mdan): Get rid of alwayslink, it's nonstandard.
    alwayslink = 1,
)

# TODO(mdan): Pull these transitive header deps in a more decent fashion.
# TODO(mdan): Get rid of headers-only lib, it's nonstandard. Use cc_shared_library?
cc_library(
    name = "runtime_client_headers",
    textual_hdrs = [
        "runtime_client.h",
        "//tensorflow/c/eager:pywrap_required_hdrs",
        "//tensorflow/core/common_runtime/eager:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/coordination:pywrap_required_hdrs",
        "//tensorflow/core/distributed_runtime/eager:pywrap_required_hdrs",
    ],
    # TODO(b/220191419): Remove this once the `runtime_client_pybind` build rule is moved to this file.
    visibility = ["//tensorflow:internal"],
)

tf_cc_test(
    name = "runtime_client_cc_test",
    srcs = ["runtime_client_test.cc"],
    deps = [
        ":runtime_client_cc",
        "//tensorflow/c:c_api_experimental",  # buildcleaner: keep (registers CPU ops?)
        "//tensorflow/c:tensor_interface",
        "//tensorflow/c/eager:immediate_execution_tensor_handle",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:op_def_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:Parser",
    ],
)

# TODO(mdan): Drop function_proto_py_pb2 once pybind11_protobuf is available.
pytype_strict_library(
    name = "runtime_client_py",
    srcs = [
        "runtime_client.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/core/framework:function_proto_py",
        "//tensorflow/python:pywrap_tensorflow",  # buildcleaner: keep (required for TF pybind)
        "//tensorflow/python:runtime_client_pybind",
    ],
)

py_strict_test(
    name = "runtime_client_py_test",
    srcs = ["runtime_client_test.py"],
    main = "runtime_client_test.py",
    python_version = "PY3",
    tags = ["no_oss"],  # b/219089812
    deps = [
        ":runtime_client_py",
        "//tensorflow/core/framework:function_proto_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:execute",
        "//tensorflow/python/platform:client_testlib",
    ],
)
