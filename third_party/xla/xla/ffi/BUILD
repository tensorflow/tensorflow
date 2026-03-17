load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl/platform:build_config.bzl", "tf_proto_library")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
)

package_group(
    name = "ffi_internal",
    packages = [
        "//xla/backends/cpu",
        "//xla/backends/gpu",
    ],
)

cc_library(
    name = "api",
    hdrs = ["//xla/ffi/api:api_headers"],
    visibility = [":ffi_internal"],
    deps = [
        "//xla/ffi/api:c_api",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "call_frame",
    srcs = ["call_frame.cc"],
    hdrs = ["call_frame.h"],
    deps = [
        ":api",
        ":attribute_map",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_address",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "call_frame_test",
    srcs = ["call_frame_test.cc"],
    deps = [
        ":attribute_map",
        ":call_frame",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/stream_executor:device_address",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "execution_context",
    srcs = ["execution_context.cc"],
    hdrs = ["execution_context.h"],
    deps = [
        ":type_registry",
        "//xla:util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:logging",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

xla_cc_test(
    name = "execution_context_test",
    srcs = ["execution_context_test.cc"],
    deps = [
        ":execution_context",
        ":type_registry",
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
        ":execution_state_proto_cc",
        ":type_registry",
        "//xla:util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

xla_cc_test(
    name = "execution_state_test",
    srcs = ["execution_state_test.cc"],
    deps = [
        ":execution_state",
        ":execution_state_proto_cc",
        ":type_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "ffi",
    hdrs = ["ffi.h"],
    deps = [
        ":api",
        ":execution_context",
        ":execution_state",
        ":type_registry",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_address",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "invoke",
    srcs = ["invoke.cc"],
    hdrs = ["invoke.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":api",
        ":call_frame",
        ":execution_context",
        ":execution_state",
        ":ffi_interop",
        ":ffi_structs",
        "//xla:executable_run_options",
        "//xla:util",
        "//xla/ffi/api:c_api",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "ffi_api",
    srcs = ["ffi_api.cc"],
    hdrs = ["ffi_api.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        ":execution_context",
        ":execution_state",
        ":ffi_internal_api",
        ":ffi_interop",
        ":ffi_registry",
        ":ffi_structs",
        ":invoke",
        ":type_registry",
        "//xla:util",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "//xla/stream_executor:stream",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "ffi_internal_api",
    srcs = ["ffi_internal_api.cc"],
    hdrs = ["ffi_internal_api.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":execution_context",
        ":execution_state",
        ":ffi_registry",
        ":ffi_structs",
        ":type_registry",
        "//xla:util",
        "//xla/ffi/api:c_api",
        "//xla/ffi/api:c_api_internal",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/concurrency:ref_count",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "ffi_interop",
    srcs = ["ffi_interop.cc"],
    hdrs = ["ffi_interop.h"],
    deps = [
        ":ffi_structs",
        "//xla/ffi/api:c_api",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "ffi_registry",
    srcs = ["ffi_registry.cc"],
    hdrs = ["ffi_registry.h"],
    deps = [
        ":api",
        ":invoke",
        "//xla:util",
        "//xla/ffi/api:c_api",
        "//xla/service:platform_util",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "ffi_structs",
    hdrs = ["ffi_structs.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":execution_context",
        ":execution_state",
        "//xla:executable_run_options",
        "//xla/hlo/ir:hlo",
        "//xla/stream_executor:device_description",
        "//xla/tsl/concurrency:async_value",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "attribute_map",
    srcs = ["attribute_map.cc"],
    hdrs = ["attribute_map.h"],
    deps = [
        ":attribute_map_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

xla_cc_test(
    name = "attribute_map_test",
    srcs = ["attribute_map_test.cc"],
    deps = [
        ":attribute_map",
        ":attribute_map_proto_cc",
        "//xla/tsl/util/proto:parse_text_proto",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_proto_library(
    name = "attribute_map_proto",
    srcs = ["attribute_map.proto"],
)

tf_proto_library(
    name = "execution_state_proto",
    srcs = ["execution_state.proto"],
)

xla_cc_test(
    name = "ffi_test",
    srcs = ["ffi_test.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    shuffle_tests = False,
    deps = [
        ":attribute_map",
        ":call_frame",
        ":execution_context",
        ":execution_state",
        ":ffi",
        ":ffi_api",  # buildcleaner: keep
        ":ffi_interop",
        ":ffi_registry",
        ":invoke",
        ":type_registry",
        "//xla:executable_run_options",
        "//xla:xla_data_proto_cc",
        "//xla/backends/cpu:ffi",
        "//xla/backends/gpu:ffi",
        "//xla/ffi/api:c_api",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:stream",
        "//xla/tsl/concurrency:async_value",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_benchmark",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_matchers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest",
        "@eigen_archive//:eigen3",
    ],
)

cc_library(
    name = "type_registry",
    srcs = ["type_registry.cc"],
    hdrs = ["type_registry.h"],
    deps = [
        "//xla:util",
        "//xla/tsl/lib/gtl:int_type",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/util:safe_reinterpret_cast",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
    ],
)

xla_cc_test(
    name = "type_registry_test",
    srcs = ["type_registry_test.cc"],
    deps = [
        ":type_registry",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)
