package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "capture_profile",
    srcs = ["capture_profile.cc"],
    hdrs = ["capture_profile.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":save_profile",
        "//tensorflow:grpc++",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "//tensorflow/core/profiler:profiler_analysis_proto_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "save_profile",
    srcs = ["save_profile.cc"],
    hdrs = ["save_profile.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)
