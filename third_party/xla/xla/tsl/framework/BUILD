# Description:
#   Tensor Standard Libraries.
#
#   The libraries in this package are not allowed to have ANY dependencies
#   to other TF components outside of TSL.

load("//xla/tsl:tsl.bzl", "if_windows", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load(
    "//xla/tsl/platform:build_config.bzl",
    "tsl_cc_test",
)
load(
    "//xla/tsl/platform:build_config_root.bzl",
    "if_static",
)
load(
    "//xla/tsl/platform:rules_cc.bzl",
    "cc_library",
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
    visibility = internal_visibility(["//tensorflow/core:__subpackages__"]),
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ] + if_static(
        extra_deps = [
            ":allocator_registry_impl",
            "@com_google_absl//absl/synchronization",
            "//xla/tsl/lib/gtl:inlined_vector",
            "@local_tsl//tsl/platform:strcat",
            "@local_tsl//tsl/platform:stringprintf",
            "//xla/tsl/platform:env",
            "//xla/tsl/platform:env_impl",
            "//xla/tsl/platform:logging",
            "//xla/tsl/platform:macros",
            "@local_tsl//tsl/platform:platform_port",
            "@local_tsl//tsl/platform:thread_annotations",
            "//xla/tsl/platform:types",
        ],
        otherwise = [
            "//xla/tsl/lib/gtl:inlined_vector",
            "//xla/tsl/platform:logging",
            "@local_tsl//tsl/platform:platform_port",
            "@local_tsl//tsl/platform:strcat",
            "//xla/tsl/platform:env",
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
    visibility = internal_visibility([
        "//xla:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "@local_tsl//tsl:__subpackages__",
        "//tensorflow/python/profiler/internal:__pkg__",
        "//third_party/xprof:__subpackages__",
    ]),
    deps = [
        ":numeric_types",
        ":type_traits",
        "//xla/tsl/lib/gtl:inlined_vector",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:strcat",
        "@local_tsl//tsl/platform:stringprintf",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/profiler/lib:scoped_memory_debug_annotation",
        "@local_tsl//tsl/profiler/lib:traceme",
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
        "//xla/tsl/lib/core:bits",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:types",
        "//xla/tsl/profiler/utils:trace_filter_utils",
        "//xla/tsl/protobuf:bfc_memory_map_proto_cc",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:numbers",
        "@local_tsl//tsl/platform:stacktrace",
        "@local_tsl//tsl/profiler/lib:scoped_memory_debug_annotation",
        "@local_tsl//tsl/profiler/lib:traceme",
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
        "//xla/tsl/lib/gtl:int_type",
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
        "//xla/tsl/lib/gtl:int_type",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:macros",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "device_id_utils",
    srcs = ["device_id_utils.cc"],
    hdrs = [
        "device_id_utils.h",
    ],
    deps = [
        ":device_id_impl",
        ":device_type",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:device_name_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:str_util",
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
    visibility = internal_visibility([
        "//tensorflow/compiler:__subpackages__",
        "//tensorflow/core:__subpackages__",
    ]),
    deps = [
        ":fixedpoint_types",
        "//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "shared_counter",
    hdrs = ["shared_counter.h"],
    features = ["parse_headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    deps = [
        "//xla/tsl/lib/monitoring:counter",
    ],
)

cc_library(
    name = "fixedpoint_types",
    hdrs = ["fixedpoint_types.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    visibility = internal_visibility([
        "//tensorflow/core/framework:__pkg__",
    ]),
    deps = [
        ":numeric_types",
        "//xla/tsl/platform:types",
    ],
)

filegroup(
    name = "cancellation_hdrs",
    srcs = [
        "cancellation.h",
    ],
    visibility = internal_visibility(["//tensorflow/core:__subpackages__"]),
)

cc_library(
    name = "cancellation",
    srcs = [
        "cancellation.cc",
    ],
    hdrs = [
        "cancellation.h",
    ],
    copts = if_windows(
        [],
        ["-Wno-thread-safety-precise"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/lib/gtl:flatmap",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:hash",
        "@local_tsl//tsl/platform:notification",
        "@local_tsl//tsl/platform:stringpiece",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "serving_device_selector",
    srcs = ["serving_device_selector.cc"],
    hdrs = ["serving_device_selector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/platform:logging",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "serving_device_selector_policies",
    srcs = ["serving_device_selector_policies.cc"],
    hdrs = ["serving_device_selector_policies.h"],
    features = ["-layering_check"],
    deps = [
        ":serving_device_selector",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "real_time_in_memory_metric",
    hdrs = ["real_time_in_memory_metric.h"],
)

tsl_cc_test(
    name = "cancellation_test",
    size = "small",
    srcs = ["cancellation_test.cc"],
    deps = [
        ":cancellation",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:env_impl",
        "//xla/tsl/platform:status",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:notification",
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
    visibility = internal_visibility([
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

tsl_cc_test(
    name = "device_id_utils_test",
    srcs = [
        "device_id_utils_test.cc",
    ],
    deps = [
        ":device_id_impl",
        ":device_id_utils",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/protobuf:error_codes_proto_impl_cc",
        "//xla/tsl/util:device_name_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

tsl_cc_test(
    name = "real_time_in_memory_metric_test",
    srcs = ["real_time_in_memory_metric_test.cc"],
    deps = [
        ":real_time_in_memory_metric",
        "//xla/tsl/platform:test",
        "@com_google_googletest//:gtest_main",
    ],
)
