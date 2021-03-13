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
    ],
    licenses = ["notice"],  # Apache 2.0
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "array_slice",
    hdrs = ["array_slice.h"],
    deps = [
        "//tensorflow/core/lib/gtl:inlined_vector",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "cleanup",
    hdrs = ["cleanup.h"],
    deps = ["//tensorflow/core/platform:macros"],
)

cc_library(
    name = "compactptrset",
    hdrs = ["compactptrset.h"],
    deps = ["//tensorflow/core/lib/gtl:flatset"],
)

cc_library(
    name = "edit_distance",
    hdrs = ["edit_distance.h"],
    deps = [
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:inlined_vector",
    ],
)

cc_library(
    name = "flatmap",
    hdrs = ["flatmap.h"],
    visibility = [
        "//tensorflow/c/eager:__pkg__",
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/framework:__pkg__",
        "//tensorflow/core/lib/histogram:__pkg__",
        "//tensorflow/core/lib/random:__pkg__",
        "//tensorflow/core/lib/strings:__pkg__",
    ],
    deps = [
        "//tensorflow/core/lib/gtl:flatrep",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "flatrep",
    hdrs = ["flatrep.h"],
    deps = [
        "//tensorflow/core/platform:prefetch",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "flatset",
    hdrs = ["flatset.h"],
    deps = [
        "//tensorflow/core/lib/gtl:flatrep",
        "//tensorflow/core/lib/hash",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "inlined_vector",
    hdrs = ["inlined_vector.h"],
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "int_type",
    hdrs = ["int_type.h"],
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:types",
    ],
)

cc_library(
    name = "iterator_range",
    hdrs = ["iterator_range.h"],
)

cc_library(
    name = "manual_constructor",
    hdrs = ["manual_constructor.h"],
    deps = [
        "//tensorflow/core/platform:macros",
        "//tensorflow/core/platform:platform_port",
    ],
)

cc_library(
    name = "map_util",
    srcs = [
        "map_util.h",
        "//tensorflow/core/lib/gtl/subtle:map_traits",
    ],
    hdrs = ["map_util.h"],
)

cc_library(
    name = "optional",
    hdrs = ["optional.h"],
    deps = ["@com_google_absl//absl/types:optional"],
)

cc_library(
    name = "priority_queue_util",
    hdrs = ["priority_queue_util.h"],
)

cc_library(
    name = "top_n",
    hdrs = ["top_n.h"],
    deps = ["//tensorflow/core/platform:logging"],
)

filegroup(
    name = "legacy_lib_gtl_headers",
    srcs = [
        "array_slice.h",
        "cleanup.h",
        "compactptrset.h",
        "edit_distance.h",
        "flatmap.h",
        "flatset.h",
        "inlined_vector.h",
        "optional.h",
        "priority_queue_util.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_internal_public_gtl_headers",
    srcs = [
        "edit_distance.h",
        "int_type.h",
        "iterator_range.h",
        "manual_constructor.h",
        "map_util.h",
        "top_n.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
        "manual_constructor.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_android_gif_internal_headers",
    srcs = [
        "cleanup.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/lib/gif:__pkg__",
    ],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "array_slice.h",
        "flatmap.h",
        "flatrep.h",
        "inlined_vector.h",
        "top_n.h",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "cleanup.h",
        "edit_distance.h",
        "flatset.h",
        "int_type.h",
        "iterator_range.h",
        "manual_constructor.h",
        "map_util.h",
        "optional.h",
        "priority_queue_util.h",
        "//tensorflow/core/lib/gtl/subtle:map_traits",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_gtl_all_headers",
    srcs = [
        "array_slice.h",
        "cleanup.h",
        "compactptrset.h",
        "edit_distance.h",
        "flatmap.h",
        "flatrep.h",
        "flatset.h",
        "inlined_vector.h",
        "int_type.h",
        "iterator_range.h",
        "manual_constructor.h",
        "map_util.h",
        "optional.h",
        "priority_queue_util.h",
        "top_n.h",
        "//tensorflow/core/lib/gtl/subtle:map_traits",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)

filegroup(
    name = "legacy_lib_gtl_tests",
    srcs = [
        "cleanup_test.cc",
        "compactptrset_test.cc",
        "edit_distance_test.cc",
        "flatmap_test.cc",
        "flatset_test.cc",
        "int_type_test.cc",
        "iterator_range_test.cc",
        "manual_constructor_test.cc",
        "map_util_test.cc",
        "top_n_test.cc",
    ],
    visibility = ["//tensorflow/core:__pkg__"],
)
