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
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
)

cc_library(
    name = "op_metrics_db_utils",
    srcs = ["op_metrics_db_utils.cc"],
    hdrs = ["op_metrics_db_utils.h"],
    deps = [
        ":math_utils",
        ":tf_op_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_utils",
    srcs = ["op_utils.cc"],
    hdrs = ["op_utils.h"],
    deps = [
        ":op_metrics_db_utils",
        ":tf_op_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tf_op_utils",
    srcs = ["tf_op_utils.cc"],
    hdrs = ["tf_op_utils.h"],
    deps = [
        "//tensorflow/core:regexp_internal",
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
    ],
)

cc_library(
    name = "timespan",
    hdrs = ["timespan.h"],
    deps = [
        ":time_utils",
        "//tensorflow/core:lib",
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
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
)

cc_library(
    name = "xplane_builder",
    srcs = ["xplane_builder.cc"],
    hdrs = ["xplane_builder.h"],
    visibility = [":friends"],
    deps = [
        ":tf_op_utils",
        ":time_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_schema",
    srcs = ["xplane_schema.cc"],
    hdrs = ["xplane_schema.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "xplane_utils",
    srcs = ["xplane_utils.cc"],
    hdrs = ["xplane_utils.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_visitor",
    hdrs = ["xplane_visitor.h"],
    visibility = [":friends"],
    deps = [
        ":time_utils",
        ":timespan",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "metadata_matcher",
    srcs = ["metadata_matcher.cc"],
    hdrs = ["metadata_matcher.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "metadata_matcher_test",
    size = "small",
    srcs = ["metadata_matcher_test.cc"],
    deps = [
        ":metadata_matcher",
        ":xplane_schema",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
