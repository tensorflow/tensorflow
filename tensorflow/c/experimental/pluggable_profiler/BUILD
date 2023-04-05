# Description:
# Profiler C API

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

filegroup(
    name = "headers",
    srcs = [
        "pluggable_profiler.h",
    ],
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "pluggable_profiler_hdrs",
    hdrs = ["pluggable_profiler.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status_headers",
    ],
)

cc_library(
    name = "pluggable_profiler",
    srcs = ["pluggable_profiler.cc"],
    hdrs = ["pluggable_profiler.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        ":pluggable_profiler_internal",
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/device:device_utils",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "pluggable_profiler_internal",
    hdrs = [
        "pluggable_profiler.h",
        "pluggable_profiler_internal.h",
    ],
    visibility = [
        "//tensorflow/c:__subpackages__",
        "//tensorflow/core/common_runtime/pluggable_device:__subpackages__",
    ],
    deps = [
        "//tensorflow/c:c_api_macros",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
)
