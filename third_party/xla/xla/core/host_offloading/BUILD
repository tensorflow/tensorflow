load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_libtpu_portable")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:build_config.bzl", _strict_cc_test_impl = "strict_cc_test")

# re-expose for convenience
strict_cc_test = _strict_cc_test_impl

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "host_offloading_allocator",
    hdrs = ["host_offloading_allocator.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_offloading_buffer",
    hdrs = ["host_offloading_buffer.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_offloading_layout_analysis",
    srcs = ["host_offloading_layout_analysis.cc"],
    hdrs = ["host_offloading_layout_analysis.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:computation_layout",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

strict_cc_test(
    name = "host_offloading_layout_analysis_test",
    srcs = ["host_offloading_layout_analysis_test.cc"],
    deps = [
        ":host_offloading_layout_analysis",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/platform:status_matchers",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "host_offloading_transforms",
    srcs = ["host_offloading_transforms.cc"],
    hdrs = ["host_offloading_transforms.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

strict_cc_test(
    name = "host_offloading_transforms_test",
    srcs = ["host_offloading_transforms_test.cc"],
    deps = [
        ":host_offloading_transforms",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "host_offloading_executable_proto",
    srcs = ["host_offloading_executable.proto"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = ["//xla/service:hlo_proto"],
)

cc_library(
    name = "host_offloading_executable",
    hdrs = ["host_offloading_executable.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":host_offloading_allocator",
        ":host_offloading_buffer",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_executable",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "host_offloading_nanort_executable",
    srcs = ["host_offloading_nanort_executable.cc"],
    hdrs = ["host_offloading_nanort_executable.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":host_offloading_buffer",
        ":host_offloading_executable",
        ":host_offloading_executable_proto_cc",
        ":host_offloading_layout_analysis",
        "//xla:debug_options_flags",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/backends/cpu/nanort:nanort_client",
        "//xla/backends/cpu/nanort:nanort_executable",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:utils",
        "//xla/runtime:device_id",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "host_offloading_pjrt_executable",
    srcs = ["host_offloading_pjrt_executable.cc"],
    hdrs = ["host_offloading_pjrt_executable.h"],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        ":host_offloading_buffer",
        ":host_offloading_executable",
        ":host_offloading_executable_proto_cc",
        ":host_offloading_layout_analysis",
        ":host_offloading_transforms",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/plugin/xla_cpu:cpu_client_options",
        "//xla/pjrt/plugin/xla_cpu:xla_cpu_pjrt_client",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

strict_cc_test(
    name = "host_offloading_executable_test",
    srcs = ["host_offloading_executable_test.cc"],
    deps = [
        ":host_offloading_buffer",
        ":host_offloading_executable",
        ":host_offloading_executable_proto_cc",
        ":host_offloading_nanort_executable",
        ":host_offloading_pjrt_executable",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla/backends/cpu/nanort:nanort_executable",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_module_config",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:casts",
    ],
)

cc_library(
    name = "annotate_host_compute_offload",
    srcs = [
        "annotate_host_compute_offload.cc",
    ],
    hdrs = [
        "annotate_host_compute_offload.h",
    ],
    compatible_with = get_compatible_with_libtpu_portable(),
    deps = [
        "//xla:side_effect_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/pass:hlo_pass",
        "//xla/service:host_offload_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/platform:statusor",
    ],
)

strict_cc_test(
    name = "annotate_host_compute_offload_test",
    srcs = ["annotate_host_compute_offload_test.cc"],
    deps = [
        ":annotate_host_compute_offload",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_module_group",
        "//xla/hlo/pass:hlo_pass_pipeline",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/hlo/testlib:verified_hlo_module",
        "//xla/service:host_offload_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
    ],
)
