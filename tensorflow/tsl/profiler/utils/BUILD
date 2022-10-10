load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")
load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//tensorflow/tsl/profiler:internal",
    ],
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
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
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

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":math_utils",
        ":timespan",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_builder_test",
    size = "small",
    srcs = ["xplane_builder_test.cc"],
    deps = [
        ":xplane_builder",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":math_utils",
        ":tf_xplane_visitor",
        ":timespan",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/tsl/platform:fingerprint",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:context_types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "xplane_utils_test",
    srcs = ["xplane_utils_test.cc"],
    deps = [
        ":math_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tf_xplane_visitor",
    hdrs = ["tf_xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "parse_annotation",
    srcs = ["parse_annotation.cc"],
    hdrs = ["parse_annotation.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "parse_annotation_test",
    srcs = ["parse_annotation_test.cc"],
    deps = [
        ":parse_annotation",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "group_events",
    srcs = ["group_events.cc"],
    hdrs = ["group_events.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/lib/gtl:map_util",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:context_types_hdrs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_test_utils",
    testonly = True,
    srcs = ["xplane_test_utils.cc"],
    hdrs = ["xplane_test_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
    ],
)

tf_cc_test(
    name = "group_events_test",
    srcs = ["group_events_test.cc"],
    deps = [
        ":group_events",
        ":tf_xplane_visitor",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_test_utils",
        ":xplane_visitor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_xplane_utils",
    srcs = ["tpu_xplane_utils.cc"],
    hdrs = ["tpu_xplane_utils.h"],
    deps = [
        ":xplane_schema",
        ":xplane_utils",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/platform:regexp",
    ],
)

tf_cc_test(
    name = "tpu_xplane_utils_test",
    srcs = ["tpu_xplane_utils_test.cc"],
    deps = [
        ":tpu_xplane_utils",
        ":xplane_schema",
        ":xplane_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ],
)
