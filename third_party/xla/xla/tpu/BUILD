# Description: StreamExecutor Interface for TPUs

load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/experimental/dtensor:__subpackages__",
        "//learning/brain/google/xla:__subpackages__",
        "//learning/brain/research/pjrt:__subpackages__",
        "//learning/brain/tfrc/executor:__subpackages__",
        "//learning/brain/tfrt/tpu_plugin:__subpackages__",
        "//tensorflow/compiler/jit:__subpackages__",
        "//tensorflow/compiler/mlir:__subpackages__",
        "//xla:__subpackages__",
        "//xla/backends/profiler/tpu:__subpackages__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "c_api_decl",
    hdrs = [
        "c_api_decl.h",
        "c_api_defn.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        "//xla/stream_executor:event",
        "//xla/stream_executor:stream",
        "//xla/stream_executor:stream_executor_h",
    ],
)

cc_library(
    name = "tpu_executor_c_api_hdrs",
    hdrs = ["tpu_executor_c_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":libtftpu_header",
    ],
)

cc_library(
    name = "c_api_conversions",
    srcs = ["c_api_conversions.cc"],
    hdrs = ["c_api_conversions.h"],
    deps = [
        ":c_api_decl",
        ":proto_helper",
        ":tpu_api",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        ":tpu_ops_c_api_hdrs",
        "//xla:executable_run_options",
        "//xla:literal",
        "//xla:shape_layout",
        "//xla:shape_tree",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:computation_layout",
        "//xla/service:computation_placer_hdr",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:maybe_owning_device_address",
        "//xla/service:shaped_buffer",
        "//xla/stream_executor:device_address",
        "//xla/stream_executor:device_address_allocator",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
    ],
)

xla_cc_test(
    name = "c_api_conversions_test",
    srcs = ["c_api_conversions_test.cc"],
    deps = [
        ":c_api_conversions",
        ":c_api_decl",
        ":proto_helper",
        "//xla:executable_run_options",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "libtftpu_header",
    hdrs = ["libtftpu.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "noncopyable_buffer",
    hdrs = ["noncopyable_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:platform_port",
    ],
)

cc_library(
    name = "status_helper",
    hdrs = ["status_helper.h"],
    deps = [
        ":c_api_decl",
        ":tpu_executor_api",
        ":tpu_executor_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tsl_status_helper",
    hdrs = ["tsl_status_helper.h"],
    deps = [
        ":c_api_decl",
        "//xla/tsl/c:tsl_status",
        "//xla/tsl/c:tsl_status_helper",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "proto_helper",
    srcs = ["proto_helper.cc"],
    hdrs = ["proto_helper.h"],
    deps = [
        ":c_api_decl",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:check",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_api_dlsym_set_fn",
    hdrs = ["tpu_api_dlsym_set_fn.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tpu_ops_c_api_hdrs",
    srcs = [],
    hdrs = [
        "tpu_ops_c_api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":libtftpu_header",
        "@com_google_absl//absl/status:statusor",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_executor_init_fns",
    hdrs = ["tpu_executor_init_fns.inc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_api",
        ":tpu_executor_api",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "tpu_library_init_fns",
    hdrs = ["tpu_library_init_fns.inc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_api",
        ":tpu_executor_init_fns",
        ":tpu_ops_c_api_hdrs",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tpu_profiler_init_fns",
    hdrs = ["tpu_profiler_init_fns.inc"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_api",
        ":tpu_profiler_c_api_hdrs",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "tpu_api",
    srcs = ["tpu_api.cc"],
    hdrs = ["tpu_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        ":tpu_ops_c_api_hdrs",
        ":tpu_profiler_c_api_hdrs",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "tpu_initialize_util",
    srcs = ["tpu_initialize_util.cc"],
    hdrs = [
        "tpu_initialize_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/platform:logging",
    ],
)

cc_library(
    name = "tpu_executor_api",
    srcs = ["tpu_executor_api.cc"],
    hdrs = ["tpu_executor_api.h"],
    deps = [
        ":libtftpu_header",
        ":tpu_executor_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_profiler_c_api_hdrs",
    hdrs = ["tpu_profiler_c_api.h"],
    deps = [
        ":c_api_decl",
        ":libtftpu_header",
    ],
)

cc_library(
    name = "tpu_topology_external",
    srcs = ["tpu_topology.cc"],
    hdrs = ["tpu_topology.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_api_decl",
        ":tpu_executor_api",
        "@tsl//tsl/platform:types",
    ],
)
