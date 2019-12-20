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
        "//tensorflow/core/profiler/convert:op_stats_to_tf_stats",
        "//tensorflow/core/profiler/convert:xplane_to_op_stats",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
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
        "//tensorflow/core/profiler/lib:profiler_session",
        "@com_google_absl//absl/strings",
    ],
)
