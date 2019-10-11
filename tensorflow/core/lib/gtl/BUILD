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
# compactptrset, flatmap, flatset, manual_constructor, + all tests.

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
    name = "edit_distance",
    hdrs = ["edit_distance.h"],
    deps = [
        "//tensorflow/core/lib/gtl:array_slice",
        "//tensorflow/core/lib/gtl:inlined_vector",
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
    deps = [],
)

cc_library(
    name = "map_util",
    srcs = [
        "map_util.h",
        "subtle/map_traits.h",
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
    deps = [],
)

cc_library(
    name = "stl_util",
    hdrs = ["stl_util.h"],
    deps = ["@com_google_absl//absl/meta:type_traits"],
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
        "stl_util.h",
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
        "stl_util.h",
        "subtle/map_traits.h",
        "top_n.h",
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
