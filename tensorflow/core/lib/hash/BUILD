load(
    "//tensorflow:tensorflow.bzl",
    "if_linux_x86_64",
    "tf_copts",
)

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
    srcs = ["hash.cc"],
    hdrs = ["hash.h"],
    deps = [
        "//tensorflow/core/lib/core:raw_coding",
        "//tensorflow/core/lib/core:stringpiece",
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
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
    name = "legacy_lib_hash_all_srcs",
    srcs = [
        "crc32c.cc",
        "crc32c_accelerate.cc",
        "hash.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_impl_srcs",
    srcs = [
        "crc32c.cc",
        "hash.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_crc32_accelerate_srcs",
    srcs = [
        "crc32c_accelerate.cc",
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
