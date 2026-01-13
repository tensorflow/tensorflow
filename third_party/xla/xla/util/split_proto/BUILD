load("@rules_cc//cc:cc_library.bzl", "cc_library")
load(
    "//xla:xla.default.bzl",
    "xla_cc_test",
)
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "split_proto",
    srcs = ["split_proto.proto"],
)

cc_library(
    name = "split_proto_reader",
    srcs = ["split_proto_reader.cc"],
    hdrs = ["split_proto_reader.h"],
    deps = [
        ":split_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/records:record_reader",
    ],
)

xla_cc_test(
    name = "split_proto_reader_test",
    srcs = ["split_proto_reader_test.cc"],
    deps = [
        ":split_proto_cc",
        ":split_proto_reader",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
        "@com_google_riegeli//riegeli/records:record_writer",
    ],
)

cc_library(
    name = "split_executable_and_options_writer",
    srcs = ["split_executable_and_options_writer.cc"],
    hdrs = ["split_executable_and_options_writer.h"],
    deps = [
        ":split_proto_cc",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:field_mask_util",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/records:record_writer",
    ],
)

xla_cc_test(
    name = "split_executable_and_options_writer_test",
    srcs = ["split_executable_and_options_writer_test.cc"],
    deps = [
        ":split_executable_and_options_writer",
        ":split_proto_reader",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)

cc_library(
    name = "split_gpu_executable_writer",
    srcs = ["split_gpu_executable_writer.cc"],
    hdrs = ["split_gpu_executable_writer.h"],
    deps = [
        ":split_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/records:record_writer",
    ],
)

xla_cc_test(
    name = "split_gpu_executable_writer_test",
    srcs = ["split_gpu_executable_writer_test.cc"],
    deps = [
        ":split_gpu_executable_writer",
        ":split_proto_reader",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)
