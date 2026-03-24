load("//xla/tsl:tsl.bzl", "if_google")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_proto_library",
    "tsl_cc_test",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "proto_utils",
    hdrs = ["proto_utils.h"],
    deps = [
        "@com_google_absl//absl/time",
    ] + if_google(["@com_google_protobuf//:duration_cc_proto"]),
)

cc_library(
    name = "proto_matchers",
    testonly = True,
    srcs = ["proto_matchers.cc"],
    hdrs = ["proto_matchers.h"],
    visibility = [
        "//learning/pathways:__subpackages__",
        "//third_party/pathways:__subpackages__",  # For OSS
        "//xla:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_for_library",
        "@tsl//tsl/platform:protobuf",
    ],
)

tf_proto_library(
    name = "proto_matchers_test_protos",
    srcs = ["proto_matchers_test_protos.proto"],
)

tsl_cc_test(
    name = "proto_matchers_test",
    srcs = ["proto_matchers_test.cc"],
    deps = [
        ":proto_matchers",
        ":proto_matchers_test_protos_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "parse_text_proto",
    testonly = True,
    hdrs = ["parse_text_proto.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:protobuf",
    ],
)

tsl_cc_test(
    name = "parse_text_proto_test",
    srcs = ["parse_text_proto_test.cc"],
    deps = [
        ":parse_text_proto",
        ":proto_matchers_test_protos_cc",
        "@com_google_googletest//:gtest_main",
    ],
)
