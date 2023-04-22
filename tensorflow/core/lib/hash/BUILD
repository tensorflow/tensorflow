load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow:tensorflow.bzl",
    "if_linux_x86_64",
    "tf_copts",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/lib/gtl uses hash functions as part of their flatset/map representation
        "//tensorflow/core/lib/gtl:__pkg__",
        # tensorflow/core/lib/io/table_builder.cc uses crc functionality
        "//tensorflow/core/lib/io:__pkg__",
        # tensorflow/core/lib/strings/proto_serialization.cc hashes proto bytes
        "//tensorflow/core/lib/strings:__pkg__",
        # tensorflow/core/platform:tracing depends on hash
        "//tensorflow/core/platform:__subpackages__",
        # tensorflow/core/framework:tensor depends on hash
        "//tensorflow/core/framework:__pkg__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "crc32c",
    srcs = [
        "crc32c.cc",
        "crc32c_accelerate.cc",
    ],
    hdrs = ["crc32c.h"],
    # -msse4.2 enables the use of crc32c compiler builtins.
    copts = tf_copts() + if_linux_x86_64(["-msse4.2"]),
    deps = [
        "//tensorflow/core/lib/core:coding",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "hash",
    hdrs = ["hash.h"],
    deps = [
        "//tensorflow/core/platform:hash",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "hash.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "crc32c.cc",
        "crc32c.h",
        "crc32c_accelerate.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_hash_all_headers",
    srcs = [
        "crc32c.h",
        "hash.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_headers",
    srcs = [
        "hash.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_hash_all_tests",
    srcs = [
        "crc32c_test.cc",
        "hash_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
