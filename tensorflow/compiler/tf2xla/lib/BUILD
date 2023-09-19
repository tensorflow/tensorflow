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
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:util",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client/lib:broadcast",
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
        "@local_xla//xla:status_macros",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client/lib:constants",
        "@local_xla//xla/client/lib:math",
    ],
)

cc_library(
    name = "scatter",
    srcs = ["scatter.cc"],
    hdrs = ["scatter.h"],
    deps = [
        ":util",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client:xla_computation",
        "@local_xla//xla/client/lib:arithmetic",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:literal",
        "@local_xla//xla:literal_util",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/client:xla_builder",
        "@local_xla//xla/client:xla_computation",
    ],
)

cc_library(
    name = "data_format",
    srcs = ["data_format.cc"],
    hdrs = ["data_format.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@local_xla//xla:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/client:xla_builder",
    ],
)
