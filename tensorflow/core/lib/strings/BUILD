load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/framework references some of the headers in this package
        "//tensorflow/core/framework:__subpackages__",
        # tensorflow/core/platform still references some of the headers in this package
        "//tensorflow/core/platform:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "base64",
    hdrs = ["base64.h"],
    deps = [
        "//tensorflow/core/platform:base64",
    ],
)

cc_library(
    name = "numbers",
    hdrs = ["numbers.h"],
    deps = ["//tensorflow/core/platform:numbers"],
)

cc_library(
    name = "ordered_code",
    srcs = ["ordered_code.cc"],
    hdrs = ["ordered_code.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:stringpiece",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "proto_serialization",
    hdrs = ["proto_serialization.h"],
    deps = [
        "//tensorflow/tsl/lib/strings:proto_serialization",
    ],
)

cc_library(
    name = "proto_text_util",
    srcs = ["proto_text_util.cc"],
    hdrs = ["proto_text_util.h"],
    deps = [
        "//tensorflow/core/lib/strings:scanner",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "scanner",
    hdrs = ["scanner.h"],
    deps = ["//tensorflow/core/platform:scanner"],
)

cc_library(
    name = "str_util",
    hdrs = ["str_util.h"],
    deps = ["//tensorflow/core/platform:str_util"],
)

cc_library(
    name = "strcat",
    hdrs = ["strcat.h"],
    deps = ["//tensorflow/core/platform:strcat"],
)

cc_library(
    name = "stringprintf",
    hdrs = ["stringprintf.h"],
    deps = ["//tensorflow/core/platform:stringprintf"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "proto_text_util.cc",
        "proto_text_util.h",
        "scanner.h",
        "str_util.h",
        "strcat.h",
        "stringprintf.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "base64.h",
        "numbers.h",
        "ordered_code.cc",
        "ordered_code.h",
        "proto_serialization.h",
        "//tensorflow/tsl/lib/strings:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_strings_all_headers",
    srcs = [
        "base64.h",
        "numbers.h",
        "ordered_code.h",
        "proto_serialization.h",
        "proto_text_util.h",
        "scanner.h",
        "str_util.h",
        "strcat.h",
        "stringprintf.h",
        "//tensorflow/tsl/lib/strings:legacy_lib_strings_all_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_strings_all_tests",
    srcs = [
        "base64_test.cc",
        "ordered_code_test.cc",
        "proto_serialization_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_proto_parsing_headers",
    srcs = [
        "numbers.h",
        "strcat.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_string_headers",
    srcs = [
        "numbers.h",
        "proto_serialization.h",
        "str_util.h",
        "strcat.h",
        "stringprintf.h",
        "//tensorflow/tsl/lib/strings:legacy_lib_string_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_string_headers",
    srcs = [
        "base64.h",
        "ordered_code.h",
        "proto_serialization.h",
        "proto_text_util.h",
        "scanner.h",
        "//tensorflow/tsl/lib/strings:legacy_lib_internal_public_string_headers",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_low_level_library_tests",
    srcs = [
        "base64_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_strings_ordered_code_test",
    srcs = [
        "ordered_code_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_strings_proto_serialization_test",
    srcs = [
        "proto_serialization_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
