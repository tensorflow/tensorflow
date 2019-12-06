package(
    default_visibility = [
        "//tensorflow/core/lib/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
        "//tensorflow/core/lib/hash:__pkg__",
        "//tensorflow/core/lib/io:__pkg__",
        "//tensorflow/core/lib/random:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
        "//tensorflow/core/platform:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are:
# base64, proto_serialization, proto_text_util, and all tests.
# Note: proto_serialization depends on target lib/hash:hash being added.

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
    name = "scanner",
    hdrs = ["scanner.h"],
    deps = ["//tensorflow/core/platform:scanner"],
)

cc_library(
    name = "string_utils",
    hdrs = [
        "numbers.h",
        "str_util.h",
        "strcat.h",
    ],
    deps = [
        "//tensorflow/core/platform:numbers",
        "//tensorflow/core/platform:str_util",
        "//tensorflow/core/platform:strcat",
    ],
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
    name = "legacy_lib_android_gif_internal_string_headers",
    srcs = [
        "numbers.h",
        "strcat.h",
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
