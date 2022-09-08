load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_client",
    srcs = ["tpu_client.cc"],
    hdrs = [
        "tpu_client.h",
    ],
    compatible_with = [],
    deps = [
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/pjrt:semaphore",
        "//tensorflow/compiler/xla/python/tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:direct_tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:grpc_tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:pod_tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:recording_tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_driver_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/core/framework:allocator",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/platform:casts",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

pybind_extension(
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
        "//tensorflow/compiler/xla/pjrt:mlir_to_hlo",
        "//tensorflow/compiler/xla/python:python_ref_manager",
        "//tensorflow/compiler/xla/python:types",
        "//tensorflow/compiler/xla/python:util",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/python:bfloat16_lib",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@llvm-project//mlir:IR",
        "@pybind11",
    ],
)

py_library(
    name = "py_tpu_client",
    srcs = [
        "tpu_client.py",
    ],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_client_extension",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//third_party/py/numpy",
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
)
