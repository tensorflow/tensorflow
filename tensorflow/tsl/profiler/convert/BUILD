load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "xla_op_utils",
    hdrs = ["xla_op_utils.h"],
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler:__subpackages__",
        "//tensorflow/core/profiler:internal",
    ],
    deps = ["@com_google_absl//absl/strings"],
)

tf_cc_test(
    name = "xla_op_utils_test",
    size = "small",
    srcs = ["xla_op_utils_test.cc"],
    deps = [
        ":xla_op_utils",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "post_process_single_host_xplane",
    srcs = ["post_process_single_host_xplane.cc"],
    hdrs = ["post_process_single_host_xplane.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/utils:xplane_schema",
        "//tensorflow/tsl/profiler/utils:xplane_utils",
    ],
)
