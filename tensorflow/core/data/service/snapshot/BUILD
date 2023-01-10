# Distributed snapshot library.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "file_utils",
    srcs = ["file_utils.cc"],
    hdrs = ["file_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.cc"],
    deps = [
        ":file_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "path_utils",
    srcs = ["path_utils.cc"],
    hdrs = ["path_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/tsl/platform:path",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "path_utils_test",
    srcs = ["path_utils_test.cc"],
    deps = [
        ":path_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "snapshot_stream_writer",
    srcs = ["snapshot_stream_writer.cc"],
    hdrs = ["snapshot_stream_writer.h"],
    deps = [
        ":file_utils",
        ":path_utils",
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:task_runner",
        "//tensorflow/core/data/service:worker_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:regexp",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "snapshot_stream_writer_test",
    size = "small",
    srcs = ["snapshot_stream_writer_test.cc"],
    deps = [
        ":path_utils",
        ":snapshot_stream_writer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/data:snapshot_utils",
        "//tensorflow/core/data:standalone",
        "//tensorflow/core/data/service:common_proto_cc",
        "//tensorflow/core/data/service:task_runner",
        "//tensorflow/core/data/service:test_util",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:path",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/protobuf:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "utils_test",
    size = "small",
    srcs = ["utils_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/tsl/platform:protobuf",
    ],
)
