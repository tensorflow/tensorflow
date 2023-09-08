load("//xla:strict.default.bzl", "py_strict_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_tsl//tsl:tsl.default.bzl", "filegroup", "tsl_pybind_extension")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_client",
    srcs = ["tpu_client.cc"],
    hdrs = [
        "tpu_client.h",
    ],
    compatible_with = [],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:semaphore",
        "//xla/python/tpu_driver",
        "//xla/python/tpu_driver:direct_tpu_driver",
        "//xla/python/tpu_driver:grpc_tpu_driver",
        "//xla/python/tpu_driver:pod_tpu_driver",
        "//xla/python/tpu_driver:recording_tpu_driver",
        "//xla/python/tpu_driver:tpu_driver_proto_cc",
        "//xla/service:computation_placer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

tsl_pybind_extension(
    name = "tpu_client_extension",
    srcs = [
        "tpu_client_extension.cc",
    ],
    copts = [
        "-fexceptions",
        "-fno-strict-aliasing",
    ],
    features = ["-use_header_modules"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_client",
        "//third_party/py/numpy:headers",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/python:python_ref_manager",
        "//xla/python:status_casters",
        "//xla/python:types",
        "//xla/python:util",
        "//xla/service:computation_placer",
        "//xla/service:hlo_graph_dumper",
        "@llvm-project//mlir:IR",
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@local_tsl//tsl/python/lib/core:ml_dtypes_lib",
        "@pybind11",
    ],
)

py_strict_library(
    name = "py_tpu_client",
    srcs = [
        "tpu_client.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_client_extension",
        "//xla/python:xla_client",
        "//xla/python:xla_extension",
    ],
)

filegroup(
    name = "header_and_client",
    srcs = glob([
        "c_api*",
        "libtpu*",
    ]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtpu",
    hdrs = ["libtpu.h"],
    visibility = ["//visibility:public"],
)
