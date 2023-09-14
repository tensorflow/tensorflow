load("//tsl:tsl.bzl", "set_external_visibility")
load("//tsl:tsl.default.bzl", "filegroup")
load(
    "//tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//tsl/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Todo(bmzhao): Remaining targets to add to this BUILD file are: all tests.

cc_library(
    name = "compactptrset",
    hdrs = ["compactptrset.h"],
    visibility = ["//visibility:public"],
    deps = [":flatset"],
)

cc_library(
    name = "flatmap",
    hdrs = ["flatmap.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":flatrep",
        "//tsl/platform:hash",
        "//tsl/platform:logging",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "flatrep",
    hdrs = ["flatrep.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:prefetch",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "flatset",
    hdrs = ["flatset.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":flatrep",
        "//tsl/platform:hash",
        "//tsl/platform:logging",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "inlined_vector",
    hdrs = ["inlined_vector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

cc_library(
    name = "int_type",
    hdrs = ["int_type.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:macros",
        "//tsl/platform:types",
    ],
)

cc_library(
    name = "iterator_range",
    hdrs = ["iterator_range.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "map_util",
    srcs = [
        "map_util.h",
        "//tsl/lib/gtl/subtle:map_traits",
    ],
    hdrs = ["map_util.h"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_gtl_headers",
    srcs = [
        "compactptrset.h",
        "flatmap.h",
        "flatset.h",
        "inlined_vector.h",
        "iterator_range.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_internal_public_gtl_headers",
    srcs = [
        "int_type.h",
        "map_util.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_test_internal_headers",
    srcs = [
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_android_gif_internal_headers",
    srcs = [
    ],
    visibility = ["//visibility:public"],
)

# Export source files needed for mobile builds, which do not use granular targets.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "flatmap.h",
        "flatrep.h",
        "inlined_vector.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "flatset.h",
        "int_type.h",
        "iterator_range.h",
        "map_util.h",
        "//tsl/lib/gtl/subtle:map_traits",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "legacy_lib_gtl_all_headers",
    srcs = [
        "compactptrset.h",
        "flatmap.h",
        "flatrep.h",
        "flatset.h",
        "inlined_vector.h",
        "int_type.h",
        "iterator_range.h",
        "map_util.h",
        "//tsl/lib/gtl/subtle:map_traits",
    ],
    visibility = ["//visibility:public"],
)

tsl_cc_test(
    name = "tsl_lib_gtl_tests",
    srcs = [
        "compactptrset_test.cc",
        "flatmap_test.cc",
        "flatset_test.cc",
        "int_type_test.cc",
        "iterator_range_test.cc",
        "map_util_test.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":compactptrset",
        ":flatmap",
        ":flatset",
        ":int_type",
        ":iterator_range",
        ":map_util",
        "//tsl/platform:hash",
        "//tsl/platform:macros",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/platform:types",
    ],
)
