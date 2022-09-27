load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
        "//tensorflow/tsl/profiler:friends",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
)

cc_library(
    name = "time_utils",
    hdrs = ["time_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":math_utils",
    ] + if_static([
        ":time_utils_impl",
    ]),
)

cc_library(
    name = "time_utils_impl",
    srcs = [
        "time_utils.cc",
        "time_utils.h",
    ],
    copts = tf_profiler_copts(),
    deps = [
        ":math_utils",
        "@com_google_absl//absl/time",
    ],
    alwayslink = True,
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":math_utils",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "timespan_test",
    srcs = ["timespan_test.cc"],
    deps = [
        ":timespan",
        "//tensorflow/core/platform:test_main",
        "//tensorflow/tsl/platform:test",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:regexp",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tf_op_utils_test",
    size = "small",
    srcs = ["tf_op_utils_test.cc"],
    deps = [
        ":tf_op_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_schema",
    srcs = ["xplane_schema.cc"],
    hdrs = ["xplane_schema.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":tf_op_utils",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:context_types_hdrs",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_visitor",
    srcs = ["xplane_visitor.cc"],
    hdrs = ["xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":timespan",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)
