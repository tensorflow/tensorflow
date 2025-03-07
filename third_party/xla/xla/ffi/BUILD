load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "api",
    hdrs = ["//xla/ffi/api:api_headers"],
    visibility = ["//visibility:private"],
    deps = ["//xla/ffi/api:c_api"],
)

cc_library(
    name = "call_frame",
    srcs = ["call_frame.cc"],
    hdrs = ["call_frame.h"],
    deps = [
        ":api",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_memory",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "call_frame_test",
    srcs = ["call_frame_test.cc"],
    deps = [
        ":call_frame",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/stream_executor:device_memory",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "execution_context",
    srcs = ["execution_context.cc"],
    hdrs = ["execution_context.h"],
    deps = [
        ":type_id_registry",
        "//xla:util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "execution_context_test",
    srcs = ["execution_context_test.cc"],
    deps = [
        ":execution_context",
        ":type_id_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "execution_state",
    srcs = ["execution_state.cc"],
    hdrs = ["execution_state.h"],
    deps = [
        ":type_id_registry",
        "//xla:util",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "execution_state_test",
    srcs = ["execution_state_test.cc"],
    deps = [
        ":execution_state",
        "//xla/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    deps = [
        ":api",
        ":execution_context",
        ":execution_state",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:scratch_allocator",
        "//xla/stream_executor:stream",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "ffi_api",
    srcs = ["ffi_api.cc"],
    hdrs = ["ffi_api.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":api",
        ":call_frame",
        ":execution_context",
        ":execution_state",
        ":type_id_registry",
        "//xla:executable_run_options",
        "//xla:util",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:device_memory_allocator",
        "//xla/stream_executor:stream",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/concurrency:ref_count",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "attribute_map",
    srcs = ["attribute_map.cc"],
    hdrs = ["attribute_map.h"],
    deps = [
        ":call_frame",
        "//xla:util",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    shuffle_tests = False,
    deps = [
        ":call_frame",
        ":execution_context",
        ":execution_state",
        ":ffi",
        ":ffi_api",
        "//xla:executable_run_options",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/stream_executor:device_memory",
        "//xla/stream_executor:stream",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:status_matchers",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_benchmark",
    ],
)

cc_library(
    name = "type_id_registry",
    srcs = ["type_id_registry.cc"],
    hdrs = ["type_id_registry.h"],
    deps = [
        "//xla:util",
        "//xla/tsl/lib/gtl:int_type",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "type_id_registry_test",
    srcs = ["type_id_registry_test.cc"],
    deps = [
        ":type_id_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
