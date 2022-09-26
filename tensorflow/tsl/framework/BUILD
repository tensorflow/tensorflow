# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow/core/platform:rules_cc.bzl",
    "cc_library",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

# Files needed for core:framework_internal_impl.
filegroup(
    name = "framework_internal_private_hdrs",
    srcs = [
        "allocator.h",
        "allocator_registry.h",
        "numeric_types.h",
        "tracking_allocator.h",
        "type_traits.h",
    ],
)

# Files needed for core:mobile_srcs_no_runtime.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "allocator.cc",
        "allocator.h",
        "allocator_registry.cc",
        "allocator_registry.h",
        "cpu_allocator_impl.cc",
        "numeric_types.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
        "type_traits.h",
    ],
)

filegroup(
    name = "allocator_hdrs",
    srcs = [
        "allocator.h",
        "allocator_registry.h",
        "numeric_types.h",
        "tracking_allocator.h",
        "type_traits.h",
    ],
    visibility = ["//tensorflow/core:__subpackages__"],
)

# Individual targets. These should be preferred over tensorflow/core:framework
# whenever possible.

# This is redundant with the "tensorflow/core:framework" target. It's useful for
# applications that want to depend on a minimal subset of TensorFlow (e.g. XLA).
cc_library(
    name = "allocator",
    srcs = [
        "allocator.cc",
        "allocator_registry.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
    ],
    hdrs = [
        "allocator.h",
    ],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":numeric_types",
        ":type_traits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ] + if_static(
        extra_deps = [
            ":allocator_registry_impl",
            "//tensorflow/tsl/lib/gtl:inlined_vector",
            "//tensorflow/tsl/platform:strcat",
            "//tensorflow/tsl/platform:stringprintf",
            "//tensorflow/tsl/platform:env",
            "//tensorflow/tsl/platform:env_impl",
            "//tensorflow/tsl/platform:logging",
            "//tensorflow/tsl/platform:macros",
            "//tensorflow/tsl/platform:mutex",
            "//tensorflow/tsl/platform:platform_port",
            "//tensorflow/tsl/platform:thread_annotations",
            "//tensorflow/tsl/platform:types",
        ],
        otherwise = [
            "//tensorflow/tsl/lib/gtl:inlined_vector",
            "//tensorflow/tsl/platform:logging",
            "//tensorflow/tsl/platform:platform_port",
            "//tensorflow/tsl/platform:strcat",
            "//tensorflow/tsl/platform:env",
        ],
    ),
    alwayslink = 1,
)

# This target will be included in libtensorflow_framework.so via the
# framework_internal_impl target.
# All other dependencies on this target need to go through if_static guard,
# as otherwise duplicate registration in the registry will cause crashes.
cc_library(
    name = "allocator_registry_impl",
    srcs = [
        "allocator.h",
        "allocator_registry.cc",
        "allocator_registry.h",
        "cpu_allocator_impl.cc",
        "tracking_allocator.h",
    ],
    visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tsl:__subpackages__",
    ],
    deps = [
        ":numeric_types",
        ":type_traits",
        "//tensorflow/core/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/tsl/lib/gtl:inlined_vector",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:stringprintf",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "numeric_types",
    hdrs = ["numeric_types.h"],
    visibility = [
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ],
    deps = [
        "//tensorflow/tsl/platform:types",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    visibility = [
        "//tensorflow/core/framework:__pkg__",
    ],
    deps = [
        ":numeric_types",
        "//tensorflow/tsl/platform:types",
    ],
)

# Export all header files for which we do not yet provide a dedicated build
# rule. This avoids breaking all the rules in tensorflow/core/BUILD.
exports_files(
    srcs = [
        "allocator_registry.h",
        "tracking_allocator.h",
    ],
    visibility = [
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/common_runtime:__pkg__",
    ],
)

# Files whose users still need to be migrated from core:framework to the
# above targets.
# TODO(gonnet): Remove these files once targets depending on them have
# been cleaned up.
exports_files(
    srcs = [
        "allocator.h",
        "numeric_types.h",
        "type_traits.h",
    ],
)
