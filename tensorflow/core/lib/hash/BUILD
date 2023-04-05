load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "//tensorflow/tsl/platform:__subpackages__",
        # tensorflow/core/framework:tensor depends on hash
        "//tensorflow/core/framework:__pkg__",
        # For dependency on hash
        "//tensorflow/cc/experimental/tf2:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "crc32c",
    hdrs = ["crc32c.h"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:cord",
        "//tensorflow/core/platform:types",
        "//tensorflow/tsl/lib/hash:crc32c",
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
        "crc32c.h",
        "//tensorflow/tsl/lib/hash:mobile_srcs_only_runtime",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_hash_all_headers",
    srcs = [
        "crc32c.h",
        "hash.h",
        "//tensorflow/tsl/lib/hash:legacy_lib_hash_all_headers",
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
