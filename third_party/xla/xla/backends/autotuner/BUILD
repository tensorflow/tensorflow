load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "codegen_backend",
    hdrs = ["codegen_backend.h"],
    deps = [
        ":backends_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_protobuf//:any_cc_proto",
    ],
)

cc_library(
    name = "autotuner",
    srcs = ["autotuner.cc"],
    hdrs = ["autotuner.h"],
    deps = [
        ":autotuner_cache_interface",
        ":codegen_backend",
        ":profiler",
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:dump",
        "//xla/service:executable",
        "//xla/service:shaped_buffer",
        "//xla/service/gpu/autotuning:autotuner_status_key",
        "//xla/stream_executor:kernel_stats",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:blocking_counter",
        "@tsl//tsl/platform:fingerprint",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:scoped_annotation",
        "@tsl//tsl/profiler/lib:traceme",
    ],
)

xla_cc_test(
    name = "autotuner_test",
    srcs = ["autotuner_test.cc"],
    deps = [
        ":autotuner",
        ":autotuner_cache_interface",
        ":backends_proto_cc",
        ":codegen_backend",
        ":profiler",
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/backends/gpu/runtime:sequential_thunk",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/testlib:hlo_hardware_independent_test_base",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:executable",
        "//xla/service:shaped_buffer",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_executable",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor:kernel_stats",
        "//xla/tsl/distributed_runtime:call_options",
        "//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/testing:temporary_directory",
        "//xla/tsl/util/proto:proto_matchers",
        "//xla/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:any_cc_proto",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)

tf_proto_library(
    name = "autotuner_cache_proto",
    srcs = ["autotuner_cache.proto"],
    deps = ["@com_google_protobuf//:any_proto"],
)

cc_library(
    name = "profiler",
    hdrs = ["profiler.h"],
    deps = [
        "//xla/service:executable",
        "//xla/service:shaped_buffer",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "autotuner_cache_interface",
    hdrs = ["autotuner_cache_interface.h"],
    deps = [
        ":autotuner_cache_proto_cc",
        ":backends_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "file_based_autotuner_cache",
    srcs = ["file_based_autotuner_cache.cc"],
    hdrs = ["file_based_autotuner_cache.h"],
    deps = [
        ":autotuner_cache_interface",
        ":autotuner_cache_proto_cc",
        ":backends_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_protobuf//:protobuf",
        "@llvm-project//llvm:Support",
        "@tsl//tsl/platform:base64",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/platform:protobuf",
    ],
)

xla_cc_test(
    name = "file_based_autotuner_cache_test",
    srcs = ["file_based_autotuner_cache_test.cc"],
    deps = [
        ":autotuner_cache_interface",
        ":autotuner_cache_proto_cc",
        ":backends_proto_cc",
        ":file_based_autotuner_cache",
        "//xla:literal_util",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/stream_executor/cuda:cuda_compute_capability",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:any_cc_proto",
        "@tsl//tsl/platform:path",
    ],
)

tf_proto_library(
    name = "backends_proto",
    srcs = ["backends.proto"],
)
