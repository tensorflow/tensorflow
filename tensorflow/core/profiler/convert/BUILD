package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "op_metrics_to_record",
    srcs = ["op_metrics_to_record.cc"],
    hdrs = ["op_metrics_to_record.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/utils:math_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)
