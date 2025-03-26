load("//xla/tsl:tsl.bzl", "if_with_tpu_support")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "tpu_tracer",
    srcs = if_with_tpu_support(["tpu_tracer.cc"]),
    copts = tf_profiler_copts(),
    visibility = ["//xla:internal"],
    deps = [
        "//xla/stream_executor/tpu:tpu_api",
        "//xla/stream_executor/tpu:tpu_api_dlsym_set_fn",
        "//xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "//xla/stream_executor/tpu:tpu_profiler_init_fns",
        "//xla/stream_executor/tpu:tsl_status_helper",
        "//xla/tsl/c:tsl_status",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)
