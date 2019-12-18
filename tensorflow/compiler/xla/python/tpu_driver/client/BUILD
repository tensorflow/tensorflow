load("//tensorflow:tensorflow.bzl", "pybind_extension")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "tpu_client",
    srcs = ["tpu_client.cc"],
    hdrs = [
        "tpu_client.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:executable_build_options",
        "//tensorflow/compiler/xla/python:local_client",
        "//tensorflow/compiler/xla/python:semaphore",
        "//tensorflow/compiler/xla/python/tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:grpc_tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:recording_tpu_driver",
        "//tensorflow/compiler/xla/python/tpu_driver:tpu_driver_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:allocator",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
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
    module_name = "tpu_client_extension",
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_client",
        "//tensorflow/compiler/xla/python:python_ref_manager",
        "//tensorflow/compiler/xla/python:types",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//third_party/py/numpy:headers",
        "//third_party/python_runtime:headers",  # buildcleaner: keep
        "@pybind11",
    ],
)

py_library(
    name = "py_tpu_client",
    srcs = [
        "tpu_client.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_client_extension",
        "//tensorflow/compiler/xla/python:xla_client",
        "//tensorflow/compiler/xla/python:xla_extension",
        "//third_party/py/numpy",
    ],
)
