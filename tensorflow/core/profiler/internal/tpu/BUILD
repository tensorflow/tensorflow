load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//tensorflow:tensorflow.bzl", "if_with_tpu_support")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_tracer",
    srcs = if_with_tpu_support(["tpu_tracer.cc"]),
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_api_dlsym_initializer",
        "//tensorflow/core/tpu:tpu_initializer_helper",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/stream_executor/tpu:status_helper",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)
