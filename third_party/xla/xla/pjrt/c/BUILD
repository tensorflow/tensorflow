load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm_is_configured",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@local_tsl//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//xla:xla.bzl",
    "xla_cc_binary",
    "xla_cc_test",
)
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "if_macos",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
)

cc_library(
    name = "pjrt_c_api_hdrs",
    hdrs = ["pjrt_c_api.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_macros",
    hdrs = [
        "pjrt_c_api_macros.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "pjrt_c_api_tpu_hdrs",
    hdrs = ["pjrt_c_api_tpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_ffi_extension_hdrs",
    hdrs = ["pjrt_c_api_ffi_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_ffi_internal",
    srcs = ["pjrt_c_api_ffi_internal.cc"],
    hdrs = ["pjrt_c_api_ffi_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//xla/ffi:execution_context",
        "//xla/ffi:type_id_registry",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu_extension_hdrs",
    hdrs = ["pjrt_c_api_gpu_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_stream_extension_hdrs",
    hdrs = ["pjrt_c_api_stream_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_profiler_extension_hdrs",
    hdrs = ["pjrt_c_api_profiler_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_custom_partitioner_extension_hdrs",
    hdrs = ["pjrt_c_api_custom_partitioner_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_layouts_extension_hdrs",
    hdrs = ["pjrt_c_api_layouts_extension.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_wrapper_impl",
    srcs = ["pjrt_c_api_wrapper_impl.cc"],
    hdrs = ["pjrt_c_api_wrapper_impl.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_layouts_extension_hdrs",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/pjrt:mlir_to_hlo",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt:pjrt_layout",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/framework:allocator",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_helpers",
    srcs = ["pjrt_c_api_helpers.cc"],
    hdrs = ["pjrt_c_api_helpers.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_profiler_extension_hdrs",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt/distributed:key_value_store_interface",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu_internal",
    srcs = ["pjrt_c_api_cpu_internal.cc"],
    hdrs = ["pjrt_c_api_cpu_internal.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt/cpu:cpu_client",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "pjrt_c_api_cpu",
    srcs = ["pjrt_c_api_cpu.cc"],
    hdrs = ["pjrt_c_api_cpu.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_cpu_internal",
        ":pjrt_c_api_hdrs",
    ],
)

# PJRT CPU plugin.
xla_cc_binary(
    name = "pjrt_c_api_cpu_plugin.so",
    linkopts = if_macos(
        [],
        [
            "-Wl,--version-script,$(location :pjrt_c_api_cpu_version_script.lds)",
            "-Wl,--no-undefined",
        ],
    ),
    linkshared = True,
    tags = [
        "noasan",
        "nomsan",
        "notsan",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_cpu",
        ":pjrt_c_api_cpu_version_script.lds",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu_internal",
    srcs = ["pjrt_c_api_gpu_internal.cc"],
    hdrs = ["pjrt_c_api_gpu_internal.h"],
    local_defines = if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_custom_partitioner_extension_hdrs",
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_ffi_internal",
        ":pjrt_c_api_gpu_extension_hdrs",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_layouts_extension_hdrs",
        ":pjrt_c_api_profiler_extension_hdrs",
        ":pjrt_c_api_stream_extension_hdrs",
        ":pjrt_c_api_wrapper_impl",
        "//xla/backends/profiler:profiler_backends",  # To register the Host Tracers for GPU Plugin.
        "//xla/backends/profiler/gpu:device_tracer",  # To register the GPU Tracers with the GPU Plugin.
        "//xla/backends/profiler/plugin:plugin_tracer_impl",
        "//xla/backends/profiler/plugin:profiler_c_api_hdrs",
        "//xla/backends/profiler/plugin:profiler_error",
        "//xla/client:local_client",
        "//xla/ffi",
        "//xla/ffi:ffi_api",
        "//xla/ffi/api:c_api",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_compiler",
        "//xla/pjrt:pjrt_device_description",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt/gpu:gpu_helpers",
        "//xla/pjrt/gpu:gpu_topology",
        "//xla/pjrt/gpu:se_gpu_pjrt_client",
        "//xla/pjrt/gpu:se_gpu_pjrt_compiler",  # To register GPU AOT compiler
        "//xla/python:custom_partition_callback",
        "//xla/python:inspect_sharding",  # To register "InspectSharding" custom partitioning handler.
        "//xla/service:compiler",
        "//xla/service:custom_call_target_registry",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "pjrt_c_api_gpu",
    srcs = ["pjrt_c_api_gpu.cc"],
    hdrs = ["pjrt_c_api_gpu.h"],
    deps = [
        ":pjrt_c_api_gpu_internal",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_macros",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/log:initialize",
        "@local_tsl//tsl/platform",
    ],
    alwayslink = 1,
)

# PJRT GPU plugin. Can be configured to be built for CUDA or ROCM.
xla_cc_binary(
    name = "pjrt_c_api_gpu_plugin.so",
    linkopts = [
        "-Wl,--version-script,$(location :pjrt_c_api_gpu_version_script.lds)",
        "-Wl,--no-undefined",
    ],
    linkshared = True,
    tags = [
        "no_mac",  # No GPU on mac.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [
        ":pjrt_c_api_gpu",
        ":pjrt_c_api_gpu_version_script.lds",
        "//xla/service:gpu_plugin",
    ] + if_cuda_is_configured([
        "//xla/stream_executor:cuda_platform",
    ]) + if_rocm_is_configured([
        "//xla/stream_executor:rocm_platform",
    ]),
)

cc_library(
    name = "pjrt_c_api_test_base",
    testonly = 1,
    srcs = ["pjrt_c_api_test_base.cc"],
    hdrs = ["pjrt_c_api_test_base.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        "//xla:shape_util",
        "//xla/client:executable_build_options",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_executable",
        "//xla/pjrt:pjrt_future",
        "//xla/service:computation_placer_hdr",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_test(
    name = "pjrt_c_api_gpu_test",
    srcs = ["pjrt_c_api_gpu_test.cc"],
    backends = ["gpu"],
    tags = if_google([
        "config-cuda-only",
    ]),
    deps = [
        ":pjrt_c_api_ffi_extension_hdrs",
        ":pjrt_c_api_gpu",
        ":pjrt_c_api_gpu_extension_hdrs",
        ":pjrt_c_api_gpu_internal",
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_test_base",
        ":pjrt_c_api_test_common",
        ":pjrt_c_api_wrapper_impl",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/client:client_library",
        "//xla/ffi:execution_context",
        "//xla/ffi:ffi_api",
        "//xla/ffi:type_id_registry",
        "//xla/ffi/api:ffi",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt:pjrt_future",
        "//xla/pjrt/distributed:in_memory_key_value_store",
        "//xla/service:custom_call_target_registry",
        "//xla/stream_executor/gpu:gpu_init",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_helpers_test",
    srcs = ["pjrt_c_api_helpers_test.cc"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_wrapper_impl",
        "//xla:shape_util",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_common",
        "//xla/pjrt/distributed:in_memory_key_value_store",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_cpu_test",
    srcs = ["pjrt_c_api_cpu_test.cc"],
    deps = [
        ":pjrt_c_api_cpu",
        ":pjrt_c_api_test_common",
        ":pjrt_c_api_wrapper_impl",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "pjrt_c_api_test_common",
    testonly = 1,
    srcs = ["pjrt_c_api_test.cc"],
    hdrs = ["pjrt_c_api_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_c_api_hdrs",
        ":pjrt_c_api_helpers",
        ":pjrt_c_api_test_base",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt:compile_options_proto_cc",
        "//xla/pjrt:pjrt_client",
        "//xla/pjrt:pjrt_future",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:status",
    ],
)
