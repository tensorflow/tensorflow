load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        # tensorflow/core:lib effectively exposes all targets under tensorflow/core/lib/**
        "//tensorflow/core:__pkg__",
        # tensorflow/core/lib/random uses on gtl:array_slice
        "//tensorflow/core/lib/random:__pkg__",
        # tensorflow/core/lib/strings:proto_serialization uses on gtl:inlined_vector
        "//tensorflow/core/lib/strings:__pkg__",
        # tensorflow/core/lib/histogram uses array_slice
        "//tensorflow/core/lib/histogram:__pkg__",
        # tensorflow/core/framework uses array_slice, map_util, and flatmap
        "//tensorflow/core/framework:__pkg__",
        # tensorflow/core/util uses array_slice, inlined_vector
        "//tensorflow/core/util:__pkg__",
        # tensorflow/core/tfrt/utils uses array_slice, inlined_vector
        "//tensorflow/core/tfrt/utils:__pkg__",
        # tensorflow/examples/custom_ops_doc/simple_hash_table uses map_util
        "//tensorflow/examples/custom_ops_doc/simple_hash_table:__pkg__",
        "//tensorflow/compiler/xla/stream_executor:__subpackages__",
        "//tensorflow/core/lib/gtl:__subpackages__",
    ],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "compactptrset",
    hdrs = ["compactptrset.h"],
    deps = ["//tensorflow/tsl/lib/gtl:flatset"],
)

cc_library(
    name = "flatrep",
    hdrs = ["flatrep.h"],
    deps = [
        "//tensorflow/tsl/platform:prefetch",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "flatset",
    hdrs = ["flatset.h"],
    deps = [
        "//tensorflow/tsl/lib/gtl:flatrep",
        "//tensorflow/tsl/platform:hash",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "inlined_vector",
    hdrs = ["inlined_vector.h"],
    deps = [
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "int_type",
    hdrs = ["int_type.h"],
    deps = [
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "iterator_range",
    hdrs = ["iterator_range.h"],
)

cc_library(
    name = "map_util",
    srcs = [
        "map_util.h",
        "//tensorflow/tsl/lib/gtl/subtle:map_traits",
    ],
    hdrs = ["map_util.h"],
)

filegroup(
    name = "legacy_lib_gtl_headers",
    srcs = [
        "compactptrset.h",
        "flatset.h",
        "inlined_vector.h",
        "iterator_range.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_internal_public_gtl_headers",
    srcs = [
        "int_type.h",
        "map_util.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

filegroup(
    name = "legacy_android_gif_internal_headers",
    srcs = [
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "flatrep.h",
        "inlined_vector.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "flatset.h",
        "int_type.h",
        "iterator_range.h",
        "map_util.h",
        "//tensorflow/tsl/lib/gtl/subtle:map_traits",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_gtl_all_headers",
    srcs = [
        "compactptrset.h",
        "flatrep.h",
        "flatset.h",
        "inlined_vector.h",
        "int_type.h",
        "iterator_range.h",
        "map_util.h",
        "//tensorflow/tsl/lib/gtl/subtle:map_traits",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)

filegroup(
    name = "legacy_lib_gtl_tests",
    srcs = [
    ],  # TODO(ddunleavy) move tests once core/platform/test.h is in tsl
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gtl:__pkg__",
    ],
)
