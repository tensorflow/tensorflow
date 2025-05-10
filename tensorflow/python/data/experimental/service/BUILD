load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "tf_py_strict_test", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

tf_python_pybind_extension(
    name = "_pywrap_server_lib",
    srcs = ["server_lib_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_server_lib.pyi",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:dispatcher_client",
        "//tensorflow/core/data/service:grpc_util",
        "//tensorflow/core/data/service:server_lib",
        "//tensorflow/core/data/service:server_lib_headers_lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/python/lib/core:pybind11_lib",
        "//tensorflow/python/lib/core:pybind11_status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

py_strict_library(
    name = "server_lib",
    srcs = ["server_lib.py"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":_pywrap_server_lib",
        ":_pywrap_utils_exp",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:pywrap_tensorflow",
        "//tensorflow/python/util:tf_export",
    ],
)

tf_py_strict_test(
    name = "server_lib_test",
    srcs = ["server_lib_test.py"],
    deps = [
        ":server_lib",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/profiler:profiler_client",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_utils_exp",
    srcs = ["utils_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_utils_exp.pyi",
    ],
    deps = [
        "//tensorflow/core/data/service:py_utils",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_snapshot_utils",
    srcs = ["snapshot_utils_wrapper.cc"],
    enable_stub_generation = True,
    pytype_srcs = [
        "_pywrap_snapshot_utils.pyi",
    ],
    deps = [
        "//tensorflow/core/data/service/snapshot:path_utils",
        "//tensorflow/python/lib/core:pybind11_lib",
        "@local_xla//third_party/python_runtime:headers",
        "@pybind11",
    ],
)

py_strict_library(
    name = "service",
    srcs = ["__init__.py"],
    deps = [
        ":server_lib",
        "//tensorflow/python/data/experimental/ops:data_service_ops",
    ],
)
