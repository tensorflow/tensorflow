# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("//tensorflow/tsl:tsl.bzl", "set_external_visibility")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//tensorflow/tsl/platform:rules_cc.bzl",
    "cc_library",
)
load(
    "//tensorflow/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "cancellation.h",
        "device_type.h",
        "fixedpoint_types.h",
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
        "device_type.h",
        "fixedpoint_types.h",
        "numeric_types.h",
        "tracking_allocator.cc",
        "tracking_allocator.h",
        "type_traits.h",
    ],
)

# Files needed for core:mobile_srcs_only_runtime.
filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "cancellation.cc",
        "cancellation.h",
        "fixedpoint_types.h",
        "metrics.cc",
        "metrics.h",
    ],
)

filegroup(
    name = "allocator_hdrs",
    srcs = [
        "allocator.h",
        "allocator_registry.h",
        "fixedpoint_types.h",
        "numeric_types.h",
        "tracking_allocator.h",
        "type_traits.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core:__subpackages__"]),
)

# Files needed for tf2xla build.
filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = ["fixedpoint_types.h"],
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
    visibility = set_external_visibility([
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tsl:__subpackages__",
    ]),
    deps = [
        ":numeric_types",
        ":type_traits",
        "//tensorflow/tsl/lib/gtl:inlined_vector",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:stringprintf",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

cc_library(
    name = "bfc_allocator",
    srcs = [
        "allocator_retry.cc",
        "allocator_retry.h",
        "bfc_allocator.cc",
    ],
    hdrs = ["bfc_allocator.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        ":allocator",
        ":metrics",
        ":shared_counter",
        "//tensorflow/tsl/lib/core:bits",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:numbers",
        "//tensorflow/tsl/platform:stacktrace",
        "//tensorflow/tsl/platform:str_util",
        "//tensorflow/tsl/platform:strcat",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/lib:scoped_memory_debug_annotation",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/protobuf:bfc_memory_map_proto_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "device_type",
    srcs = ["device_type.cc"],
    hdrs = ["device_type.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "device_id",
    textual_hdrs = [
        "device_id.h",
        "device_id_manager.h",
    ],
    deps = [
        "//tensorflow/tsl/platform:types",
    ] + if_static([
        ":device_id_impl",
    ]),
)

cc_library(
    name = "device_id_impl",
    srcs = ["device_id_manager.cc"],
    hdrs = [
        "device_id.h",
        "device_id_manager.h",
    ],
    deps = [
        ":device_type",
        "//tensorflow/tsl/lib/gtl:int_type",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:types",
    ],
)

filegroup(
    name = "device_runtime_headers",
    srcs = [
        "device_id.h",
        "device_id_manager.h",
    ],
)

cc_library(
    name = "numeric_types",
    hdrs = ["numeric_types.h"],
    visibility = set_external_visibility([
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ]),
    deps = [
        ":fixedpoint_types",
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "shared_counter",
    hdrs = ["shared_counter.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:types",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//tensorflow/tsl/lib/monitoring:counter",
    ],
)

cc_library(
    name = "fixedpoint_types",
    hdrs = ["fixedpoint_types.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    visibility = set_external_visibility([
        "//tensorflow/core/framework:__pkg__",
    ]),
    deps = [
        ":numeric_types",
        "//tensorflow/tsl/platform:types",
    ],
)

filegroup(
    name = "cancellation_hdrs",
    srcs = [
        "cancellation.h",
    ],
    visibility = set_external_visibility(["//tensorflow/core:__subpackages__"]),
)

cc_library(
    name = "cancellation",
    srcs = [
        "cancellation.cc",
    ],
    hdrs = [
        "cancellation.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/lib/gtl:flatmap",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:hash",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:notification",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:stringpiece",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "@com_google_absl//absl/memory",
    ],
)

tsl_cc_test(
    name = "cancellation_test",
    size = "small",
    srcs = ["cancellation_test.cc"],
    deps = [
        ":cancellation",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:env_impl",
        "//tensorflow/tsl/platform:notification",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

# Export all header files for which we do not yet provide a dedicated build
# rule. This avoids breaking all the rules in tensorflow/core/BUILD.
exports_files(
    srcs = [
        "allocator_registry.h",
        "allocator_retry.cc",
        "allocator_retry.h",
        "bfc_allocator.cc",
        "bfc_allocator.h",
        "device_type.h",
        "metrics.h",
        "shared_counter.h",
        "tracking_allocator.h",
    ],
    visibility = set_external_visibility([
        "//tensorflow/core:__pkg__",
        "//tensorflow/core/common_runtime:__pkg__",
        "//tensorflow/core/common_runtime/gpu:__pkg__",
        "//tensorflow/core/framework:__pkg__",
    ]),
)

# Files whose users still need to be migrated from core:framework to the
# above targets.
# TODO(gonnet): Remove these files once targets depending on them have
# been cleaned up.
exports_files(
    srcs = [
        "allocator.h",
        "cancellation.h",
        "numeric_types.h",
        "type_traits.h",
    ],
)
