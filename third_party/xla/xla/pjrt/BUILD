# Placeholder: load py_proto_library
load("//xla:xla.bzl", "xla_cc_test")
load(
    "@local_tsl//tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
        "//xla:internal",
    ],
    packages = [
        "//third_party/australis/...",
        "//third_party/openxla_pjrt_plugin/...",
        "//third_party/py/jax/...",
    ],
)

cc_library(
    name = "worker_thread",
    srcs = ["worker_thread.cc"],
    hdrs = ["worker_thread.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "event_pool",
    srcs = ["event_pool.cc"],
    hdrs = ["event_pool.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:types",
        "//xla/stream_executor",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "semaphore",
    srcs = ["semaphore.cc"],
    hdrs = ["semaphore.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:types",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "semaphore_test",
    srcs = ["semaphore_test.cc"],
    deps = [
        ":semaphore",
        "//xla:test",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "tracked_device_buffer",
    srcs = ["tracked_device_buffer.cc"],
    hdrs = ["tracked_device_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_pool",
        ":local_device_state",
        ":utils",
        "//xla:shape_util",
        "//xla:types",
        "//xla/runtime:async_runtime",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/stream_executor",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "tracked_device_buffer_test",
    srcs = ["tracked_device_buffer_test.cc"],
    deps = [
        ":tracked_device_buffer",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:test",
        "//xla/client:client_library",
        "//xla/service:cpu_plugin",
        "//xla/stream_executor:device_memory_allocator",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "local_device_state",
    srcs = ["local_device_state.cc"],
    hdrs = ["local_device_state.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_pool",
        ":semaphore",
        ":worker_thread",
        "//xla:status",
        "//xla:util",
        "//xla/client:local_client",
        "//xla/stream_executor",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "pjrt_api",
    srcs = ["pjrt_api.cc"],
    hdrs = ["pjrt_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla:statusor",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "pjrt_api_test",
    srcs = ["pjrt_api_test.cc"],
    deps = [
        ":pjrt_api",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_wrapper_impl",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "pjrt_client",
    srcs = ["pjrt_client.cc"],
    hdrs = ["pjrt_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_common",
        ":pjrt_compiler",
        ":pjrt_device_description",
        ":pjrt_executable",
        ":pjrt_future",
        ":utils",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_cost_analysis",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/platform:errors",
    ],
)

cc_library(
    name = "pjrt_client_test_common",
    testonly = 1,
    srcs = ["pjrt_client_test.cc"],
    hdrs = ["pjrt_client_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_client",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client:xla_computation",
        "//xla/service:hlo_parser",
        "//xla/tests:literal_test_util",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "pjrt_client_test_cpu",
    srcs = ["pjrt_client_test_cpu.cc"],
    deps = [
        ":pjrt_client_test_common",
        ":tfrt_cpu_pjrt_client",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "pjrt_executable",
    srcs = ["pjrt_executable.cc"],
    hdrs = ["pjrt_executable.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":compile_options_proto_cc",
        ":execute_options_proto_cc",
        ":pjrt_common",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:computation_layout",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "pjrt_executable_test",
    srcs = ["pjrt_executable_test.cc"],
    deps = [
        ":compile_options_proto_cc",
        ":pjrt_executable",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "pjrt_device_description",
    hdrs = ["pjrt_device_description.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_common",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "pjrt_compiler",
    srcs = ["pjrt_compiler.cc"],
    hdrs = ["pjrt_compiler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":metrics",
        ":pjrt_device_description",
        ":pjrt_executable",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:fingerprint",
    ],
)

xla_cc_test(
    name = "pjrt_compiler_test",
    srcs = ["pjrt_compiler_test.cc"],
    deps = [
        ":metrics",
        ":pjrt_client",
        ":pjrt_compiler",
        ":pjrt_device_description",
        "//xla:statusor",
        "//xla/client:xla_computation",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/monitoring:cell_reader",
        "@local_tsl//tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "pjrt_common",
    hdrs = ["pjrt_common.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_placer",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    hdrs = ["metrics.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor",
        "//xla/stream_executor/gpu:gpu_init",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/lib/monitoring:counter",
        "@local_tsl//tsl/lib/monitoring:gauge",
    ],
)

tf_proto_library(
    name = "stream_executor_executable_proto",
    srcs = ["stream_executor_executable.proto"],
    cc_api_version = 2,
    protodeps = [
        ":compile_options_proto",
        "//xla:xla_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "stream_executor_executable",
    srcs = ["stream_executor_executable.cc"],
    hdrs = ["stream_executor_executable.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_executable",
        ":stream_executor_executable_proto_cc",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "pjrt_stream_executor_client",
    srcs = ["pjrt_stream_executor_client.cc"],
    hdrs = ["pjrt_stream_executor_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":event_pool",
        ":local_device_state",
        ":metrics",
        ":mlir_to_hlo",
        ":pjrt_client",
        ":pjrt_executable",
        ":pjrt_future",
        ":tracked_device_buffer",
        ":transpose",
        ":utils",
        "//xla:cpu_function_runtime",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:local_client",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/distributed:protocol_proto_cc",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer",
        "//xla/service:executable",
        "//xla/service:generic_transfer_manager",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:maybe_owning_device_memory",
        "//xla/service:shaped_buffer",
        "//xla/service:transfer_manager",
        "//xla/service/gpu:gpu_executable_run_options",
        "//xla/stream_executor",
        "//xla/stream_executor/host:host_platform_id",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "pjrt_stream_executor_client_test",
    srcs = ["pjrt_stream_executor_client_test.cc"],
    deps = [
        ":pjrt_client",
        ":pjrt_future",
        ":pjrt_stream_executor_client",
        "//xla:literal",
        "//xla:literal_comparison",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/client:client_library",
        "//xla/client:xla_builder",
        "//xla/service:cpu_plugin",
        "//xla/service:platform_util",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "interpreter_device",
    srcs = ["interpreter_device.cc"],
    hdrs = ["interpreter_device.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_stream_executor_client",
        "//xla:statusor",
        "//xla/client:client_library",
        "//xla/service:interpreter_plugin",
        "//xla/service:platform_util",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mlir_to_hlo",
    srcs = ["mlir_to_hlo.cc"],
    hdrs = ["mlir_to_hlo.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:status",
        "//xla/client:xla_computation",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:chlo_ops",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "pjrt_future",
    hdrs = ["pjrt_future.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/functional:any_invocable",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/concurrency:ref_count",
    ],
)

cc_library(
    name = "tracked_tfrt_cpu_device_buffer",
    srcs = ["tracked_tfrt_cpu_device_buffer.cc"],
    hdrs = ["tracked_tfrt_cpu_device_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:cpu_function_runtime",
        "//xla:shape_util",
        "//xla:util",
        "//xla/runtime:cpu_event",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:platform_port",
    ],
)

xla_cc_test(
    name = "tracked_tfrt_cpu_device_buffer_test",
    srcs = ["tracked_tfrt_cpu_device_buffer_test.cc"],
    deps = [
        ":tracked_tfrt_cpu_device_buffer",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/platform:env",
    ],
)

cc_library(
    name = "abstract_tfrt_cpu_buffer",
    srcs = ["abstract_tfrt_cpu_buffer.cc"],
    hdrs = ["abstract_tfrt_cpu_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_client",
        ":pjrt_future",
        ":tracked_tfrt_cpu_device_buffer",
        ":transpose",
        ":utils",
        "//xla:cpu_function_runtime",
        "//xla:literal",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/runtime:cpu_event",
        "//xla/service:shaped_buffer",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/cpu:cpu_xfeed",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

cc_library(
    name = "tfrt_cpu_pjrt_client",
    srcs = ["tfrt_cpu_pjrt_client.cc"],
    hdrs = ["tfrt_cpu_pjrt_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":abstract_tfrt_cpu_buffer",
        ":compile_options_proto_cc",
        ":mlir_to_hlo",
        ":pjrt_client",
        ":pjrt_executable",
        ":pjrt_future",
        ":semaphore",
        ":tracked_tfrt_cpu_device_buffer",
        ":transpose",
        ":utils",
        "//xla:array",
        "//xla:debug_options_flags",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:executable_build_options",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/runtime:cpu_event",
        "//xla/service:buffer_assignment",
        "//xla/service:compiler",
        "//xla/service:computation_placer_hdr",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_module_util",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_value",
        "//xla/service/cpu:buffer_desc",
        "//xla/service/cpu:cpu_compiler",
        "//xla/service/cpu:cpu_executable",
        "//xla/service/cpu:cpu_xfeed",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:dynamic_annotations",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@eigen_archive//:eigen3",  # TODO(zhangqiaorjc): Remove if use TFRT threadpool.
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/concurrency:async_value",
        "@local_tsl//tsl/concurrency:ref_count",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:denormal",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:setround",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
        "@local_tsl//tsl/profiler/lib:context_types_hdrs",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "tfrt_cpu_pjrt_client_test",
    srcs = ["tfrt_cpu_pjrt_client_test.cc"],
    deps = [
        ":tfrt_cpu_pjrt_client",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla/service:custom_call_status_public_headers",
        "//xla/service:custom_call_target_registry",
        "//xla/service:hlo_parser",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "lru_cache",
    hdrs = ["lru_cache.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:node_hash_map",
        "@local_tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "lru_cache_test",
    srcs = ["lru_cache_test.cc"],
    deps = [
        ":lru_cache",
        "//xla:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "transpose",
    srcs = [
        "transpose.cc",
        "transpose_kernels.h",
    ],
    hdrs = ["transpose.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":lru_cache",
        "//xla:permutation_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "transpose_test",
    srcs = ["transpose_test.cc"],
    deps = [
        ":transpose",
        "//xla:array",
        "//xla:permutation_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:util",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/numeric:int128",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test_benchmark",
        "@local_tsl//tsl/platform:test_main",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "pjrt_c_api_client",
    srcs = ["pjrt_c_api_client.cc"],
    hdrs = ["pjrt_c_api_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":compile_options_proto_cc",
        ":pjrt_api",
        ":pjrt_client",
        ":pjrt_common",
        ":pjrt_compiler",
        ":pjrt_device_description",
        ":pjrt_executable",
        ":pjrt_future",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:status",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/client:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/pjrt/c:pjrt_c_api_hdrs",
        "//xla/pjrt/c:pjrt_c_api_helpers",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/framework:allocator",
        "@local_tsl//tsl/platform:casts",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:fingerprint",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@stablehlo//:register",
    ],
)

xla_cc_test(
    name = "pjrt_c_api_client_test",
    srcs = ["pjrt_c_api_client_test.cc"],
    tags = [
        "nomsan",
    ],
    deps = [
        ":pjrt_api",
        ":pjrt_c_api_client",
        ":pjrt_client",
        ":pjrt_compiler",
        ":pjrt_executable",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/client:xla_builder",
        "//xla/pjrt/c:pjrt_c_api_cpu_internal",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "tf_pjrt_client",
    srcs = ["tf_pjrt_client.cc"],
    hdrs = ["tf_pjrt_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_client",
        ":pjrt_future",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "tf_pjrt_client_test",
    srcs = ["tf_pjrt_client_test.cc"],
    deps = [
        ":tf_pjrt_client",
        ":tfrt_cpu_pjrt_client",
        "//xla:literal_util",
        "//xla/service:hlo_parser",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "host_callback",
    srcs = ["host_callback.cc"],
    hdrs = ["host_callback.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_client",
        ":pjrt_future",
    ],
)

xla_cc_test(
    name = "host_callback_test",
    srcs = ["host_callback_test.cc"],
    deps = [
        ":host_callback",
        ":pjrt_client",
        "//xla/tests:literal_test_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

tf_proto_library(
    name = "compile_options_proto",
    srcs = ["compile_options.proto"],
    cc_api_version = 2,
    protodeps = [
        "//xla:xla_data_proto",
        "//xla:xla_proto",
        "//xla/stream_executor:device_description_proto",
    ],
    visibility = ["//visibility:public"],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "compile_options_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":compile_options_proto"],
# )
# copybara:uncomment_end

tf_proto_library(
    name = "execute_options_proto",
    srcs = ["execute_options.proto"],
    visibility = ["//visibility:public"],
)
