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
        ":backend_config_proto_cc",
        ":backends_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "hlo_extractor",
    srcs = ["hlo_extractor.cc"],
    hdrs = ["hlo_extractor.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/tsl/util:sorted_range",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:fingerprint",
    ],
)

cc_library(
    name = "codegen_orchestrator",
    srcs = ["codegen_orchestrator.cc"],
    hdrs = ["codegen_orchestrator.h"],
    deps = [
        ":backends_proto_cc",
        ":codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/stream_executor:kernel_stats",
        "//xla/tsl/concurrency:executor",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "config_assigner",
    srcs = ["config_assigner.cc"],
    hdrs = ["config_assigner.h"],
    deps = [
        ":autotuner_cache_interface",
        ":codegen_backend",
        ":codegen_orchestrator",
        ":config_selector",
        ":hlo_extractor",
        ":profiler",
        ":tuner",
        "//xla:status_macros",
        "//xla/hlo/ir:hlo",
        "//xla/pjrt/distributed:key_value_store_interface",
        "//xla/service:dump",
        "//xla/service/gpu/autotuning:autotuner_status_key",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/concurrency:future",
        "//xla/tsl/lib/math:math_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:status_macros",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:vlog_is_on",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
        "@tsl//tsl/platform:fingerprint",
    ],
)

# TODO(b/519057668): Move unit tests to their respective cc_library targets.
xla_cc_test(
    name = "config_assigner_test",
    srcs = ["config_assigner_test.cc"],
    deps = [
        ":autotuner_cache_interface",
        ":backends_proto_cc",
        ":codegen_backend",
        ":codegen_orchestrator",
        ":config_assigner",
        ":profiler",
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla/backends/gpu/runtime:thunk",
        "//xla/backends/gpu/runtime:thunk_executor",
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
        "//xla/tsl/platform:test",
        "//xla/tsl/testing:temporary_directory",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
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

cc_library(
    name = "profiler",
    hdrs = ["profiler.h"],
    deps = [
        "//xla/service:executable",
        "//xla/service:shaped_buffer",
        "//xla/tsl/platform:status_macros",
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
        ":backend_config_proto_cc",
        ":backends_proto_cc",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "tuner",
    srcs = ["tuner.cc"],
    hdrs = ["tuner.h"],
    deps = [
        ":codegen_backend",
        ":codegen_orchestrator",
        ":profiler",
        "//xla:autotune_results_proto_cc",
        "//xla:autotuning_proto_cc",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:shaped_buffer",
        "//xla/tsl/platform:status_macros",
        "//xla/tsl/util/proto:proto_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:any_cc_proto",
    ],
)

cc_library(
    name = "config_selector",
    srcs = ["config_selector.cc"],
    hdrs = ["config_selector.h"],
    deps = [
        ":tuner",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_proto_library(
    name = "backends_proto",
    srcs = ["backends.proto"],
)

tf_proto_library(
    name = "backend_config_proto",
    srcs = ["backend_config.proto"],
    deps = [
        "//xla:autotuning_proto",
        "//xla/service/cpu:backend_config_proto",
        "//xla/service/gpu:backend_configs",
        "//xla/tsl/protobuf:dnn_proto",
    ],
)

tf_proto_library(
    name = "autotuning_proto",
    srcs = ["autotuning.proto"],
    deps = [
        ":backend_config_proto",
        ":backends_proto",
        "@com_google_protobuf//:duration_proto",
    ],
)
