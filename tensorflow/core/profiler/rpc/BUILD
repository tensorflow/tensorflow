package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "profiler_service_impl",
    srcs = ["profiler_service_impl.cc"],
    hdrs = ["profiler_service_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow:grpc++",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_lib",
        "//tensorflow/core/profiler/lib:profiler_session",
    ],
    alwayslink = True,
)

cc_library(
    name = "profiler_server",
    srcs = ["profiler_server.cc"],
    hdrs = ["profiler_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":profiler_service_impl",
        "//tensorflow:grpc++",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_lib",
        "//tensorflow/core/profiler/lib:profiler_session",
    ],
)
