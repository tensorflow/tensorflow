# Utilities for building XLA computations.

load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/tf2xla:friends"],
    licenses = ["notice"],
)

# Filegroup used to collect source files for dependency checking.
filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

cc_library(
    name = "broadcast",
    srcs = ["broadcast.cc"],
    hdrs = ["broadcast.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder/lib:broadcast",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "random",
    srcs = ["random.cc"],
    hdrs = ["random.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:statusor",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder/lib:constants",
        "@xla//xla/hlo/builder/lib:math",
    ],
)

cc_library(
    name = "scatter",
    srcs = ["scatter.cc"],
    hdrs = ["scatter.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@xla//xla:shape_util",
        "@xla//xla:status_macros",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/hlo/builder:xla_builder",
        "@xla//xla/hlo/builder:xla_computation",
    ],
)

cc_library(
    name = "data_format",
    srcs = ["data_format.cc"],
    hdrs = ["data_format.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/status:statusor",
        "@xla//xla:shape_util",
        "@xla//xla:util",
        "@xla//xla/hlo/builder:xla_builder",
    ],
)
