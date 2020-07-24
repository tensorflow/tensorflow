load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependency")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "capture_profile",
    srcs = ["capture_profile.cc"],
    hdrs = ["capture_profile.h"],
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
    deps = [
        ":profiler_client",
        ":save_profile",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_analysis_proto_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/convert:xplane_to_profile_response",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "save_profile",
    srcs = ["save_profile.cc"],
    hdrs = ["save_profile.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "profiler_client",
    srcs = ["profiler_client.cc"],
    hdrs = ["profiler_client.h"],
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_analysis_proto_cc",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        tf_grpc_cc_dependency(),
    ],
)
