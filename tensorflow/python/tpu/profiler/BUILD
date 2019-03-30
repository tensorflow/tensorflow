licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "profiler",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tpu_profiler_analysis_pb2_grpc",
        "//tensorflow/core/profiler:profiler_analysis_proto_py",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "tpu_profiler_analysis_pb2_grpc",
    srcs = ["tpu_profiler_analysis_pb2_grpc.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/profiler:profiler_analysis_proto_py"],
)
