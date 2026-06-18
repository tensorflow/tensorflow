load(
    "//xla:xla.default.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:private"],
    licenses = ["notice"],
)

cc_library(
    name = "split_proto_cli_lib",
    srcs = ["split_proto_cli_lib.cc"],
    hdrs = ["split_proto_cli_lib.h"],
    deps = [
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util:fixed_option_set_flag",
        "//xla/util/split_proto:split_executable_and_options_writer",
        "//xla/util/split_proto:split_gpu_executable_writer",
        "//xla/util/split_proto:split_proto_cc",
        "//xla/util/split_proto:split_proto_reader",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/messages:parse_message",
        "@riegeli//riegeli/messages:serialize_message",
        "@riegeli//riegeli/records:record_reader",
    ],
)

xla_cc_binary(
    name = "split_proto_cli",
    srcs = ["split_proto_cli.cc"],
    deps = [
        ":split_proto_cli_lib",
        "//xla/service:riegeli_file_reader_factory",
        "//xla/service:riegeli_file_writer_factory",
        "//xla/tsl/util:command_line_flags",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:std_io",
        "@riegeli//riegeli/bytes:writer",
        "@tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "split_proto_cli_test",
    srcs = ["split_proto_cli_test.cc"],
    deps = [
        ":split_proto_cli_lib",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/service/gpu:dense_data_intermediate_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
        "@riegeli//riegeli/base:initializer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/bytes:writer",
    ],
)
