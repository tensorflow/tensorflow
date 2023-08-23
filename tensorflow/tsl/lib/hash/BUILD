load("//tensorflow/tsl:tsl.bzl", "set_external_visibility")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load(
    "//tensorflow/tsl:tsl.bzl",
    "if_linux_x86_64",
    "tsl_copts",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = set_external_visibility([
        # tensorflow/tsl/lib/io/table_builder.cc uses crc functionality
        "//tensorflow/tsl/lib/io:__pkg__",
        # tensorflow/core/lib/hash aliases hash for now
        "//tensorflow/core/lib/hash:__pkg__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "crc32c",
    srcs = [
        "crc32c.cc",
        "crc32c_accelerate.cc",
    ],
    hdrs = ["crc32c.h"],
    # -msse4.2 enables the use of crc32c compiler builtins.
    copts = tsl_copts() + if_linux_x86_64(["-msse4.2"]),
    deps = [
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:cord",
        "//tensorflow/tsl/platform:raw_coding",
        "//tensorflow/tsl/platform:types",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "crc32c.cc",
        "crc32c.h",
        "crc32c_accelerate.cc",
    ],
    visibility = set_external_visibility(["//tensorflow/core/lib/hash:__pkg__"]),
)

filegroup(
    name = "legacy_lib_hash_all_headers",
    srcs = [
        "crc32c.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core/lib/hash:__pkg__"]),
)

tsl_cc_test(
    name = "crc32c_test",
    size = "small",
    srcs = ["crc32c_test.cc"],
    deps = [
        ":crc32c",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
    ],
)
