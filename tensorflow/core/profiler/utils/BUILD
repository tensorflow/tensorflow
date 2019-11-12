load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
)

cc_library(
    name = "event_span",
    srcs = ["event_span.cc"],
    hdrs = ["event_span.h"],
    deps = [
        ":timespan",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
    visibility = [
        "//perftools/accelerators/xprof/convert:__pkg__",
        "//perftools/gputools/profiler/collector:__pkg__",
    ],
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    deps = [
        ":time_utils",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
    ],
)

tf_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    deps = [
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)
