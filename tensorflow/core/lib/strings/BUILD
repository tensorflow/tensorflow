package(
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/platform still references some of the headers in this package
        "//tensorflow/core/platform:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "base64",
    srcs = ["base64.cc"],
    hdrs = ["base64.h"],
    deps = [
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
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
    srcs = ["proto_serialization.cc"],
    hdrs = ["proto_serialization.h"],
    deps = [
        "//tensorflow/core/lib/gtl:inlined_vector",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:protobuf",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "proto_text_util",
    srcs = ["proto_text_util.cc"],
    hdrs = ["proto_text_util.h"],
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/platform:scanner",
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
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_strings_all_srcs",
    srcs = [
        "base64.cc",
        "ordered_code.cc",
        "proto_serialization.cc",
        "proto_text_util.cc",
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
