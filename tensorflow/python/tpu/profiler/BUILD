package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "profiler",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":profiler_analysis_pb2_grpc",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/core/profiler:profiler_analysis_proto_py",
        "//tensorflow/core/profiler:protos_all_py",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "profiler_analysis_pb2_grpc",
    srcs = ["profiler_analysis_pb2_grpc.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/profiler:profiler_analysis_proto_py"],
)
