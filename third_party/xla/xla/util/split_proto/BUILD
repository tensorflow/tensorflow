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
    default_visibility = ["//xla:__subpackages__"],
    licenses = ["notice"],
)

tf_proto_library(
    name = "split_proto",
    srcs = ["split_proto.proto"],
    visibility = [
        # This proto is an implementation detail of the split proto format,
        # it shouldn't be used outside of this package.
        "//visibility:private",
    ],
)

cc_library(
    name = "split_proto_reader",
    srcs = ["split_proto_reader.cc"],
    hdrs = ["split_proto_reader.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":split_proto_cc",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/records:record_position",
        "@riegeli//riegeli/records:record_reader",
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
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
        "@riegeli//riegeli/records:record_reader",
        "@riegeli//riegeli/records:record_writer",
    ],
)

cc_library(
    name = "split_executable_and_options_writer",
    srcs = ["split_executable_and_options_writer.cc"],
    hdrs = ["split_executable_and_options_writer.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":split_proto_cc",
        ":split_proto_riegeli_options",
        "//xla/pjrt/proto:compile_options_proto_cc",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/status",
        "@com_google_protobuf//:field_mask_util",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/records:record_writer",
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
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
    ],
)

cc_library(
    name = "split_gpu_executable_writer",
    srcs = ["split_gpu_executable_writer.cc"],
    hdrs = ["split_gpu_executable_writer.h"],
    deps = [
        ":split_proto_cc",
        ":split_proto_riegeli_options",
        "//xla/service/gpu:gpu_executable_proto_cc",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/status",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/records:record_writer",
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
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:string_writer",
    ],
)

cc_library(
    name = "split_proto_riegeli_options",
    hdrs = ["split_proto_riegeli_options.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = [
        "//visibility:private",
    ],
    deps = ["@riegeli//riegeli/records:record_writer"],
)
