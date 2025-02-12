load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

cc_library(
    name = "next_pluggable_device_allocator",
    srcs = ["next_pluggable_device_allocator.cc"],
    hdrs = ["next_pluggable_device_allocator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":next_pluggable_device_api",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
    ],
)

cc_library(
    name = "next_pluggable_device",
    srcs = [
        "next_pluggable_device.cc",
        "next_pluggable_device_context.cc",
    ],
    hdrs = [
        "next_pluggable_device.h",
        "next_pluggable_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":next_pluggable_device_allocator",
        ":next_pluggable_device_api",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/compiler/jit:pjrt_base_device",
        "//tensorflow/compiler/jit:pjrt_device_context",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "//tensorflow/core/framework:device_attributes_proto_cc",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:pjrt_state",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "next_pluggable_device_api",
    srcs = ["next_pluggable_device_api.cc"],
    hdrs = ["next_pluggable_device_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla/tsl/c:tsl_status_internal",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "next_pluggable_device_factory",
    srcs = [
        "next_pluggable_device_factory.cc",
    ],
    hdrs = [
        "next_pluggable_device_factory.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":next_pluggable_device",
        ":next_pluggable_device_api",
        ":utils",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/tsl/framework:device_id_utils",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "next_pluggable_device_factory_hdrs",
    hdrs = ["next_pluggable_device_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":next_pluggable_device_api",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "plugin_resource",
    srcs = [
        "plugin_resource.cc",
    ],
    hdrs = [
        "plugin_resource.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/framework:resource_base",
    ],
)

cc_library(
    name = "plugin_op_kernel",
    hdrs = ["plugin_op_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:bfloat16",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/status",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "direct_plugin_op_kernel",
    srcs = ["direct_plugin_op_kernel.cc"],
    hdrs = ["direct_plugin_op_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":direct_plugin_variable",
        ":plugin_coordination_service_agent_helper",
        ":plugin_op_kernel",
        ":plugin_resource",
        ":plugin_variable",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:status",
    ],
)

bool_flag(
    name = "tf_npd_use_c_api",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "tf_npd_use_c_api_enabled",
    flag_values = {":tf_npd_use_c_api": "True"},
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    defines = select({
        "//tensorflow/core/common_runtime/next_pluggable_device:tf_npd_use_c_api_enabled": [
            "DEFAULT_TF_NEXT_PLUGGABLE_DEVICE_USE_C_API=true",
        ],
        "//conditions:default": [
            "DEFAULT_TF_NEXT_PLUGGABLE_DEVICE_USE_C_API=false",
        ],
    }),
    visibility = ["//visibility:public"],
    deps = ["@com_google_absl//absl/flags:flag"],
    alwayslink = 1,
)

cc_library(
    name = "c_plugin_op_kernel",
    srcs = ["c_plugin_op_kernel.cc"],
    hdrs = ["c_plugin_op_kernel.h"],
    copts = ["-DTF_CAPI_WEAK"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_plugin_variable",
        ":plugin_coordination_service_agent",
        ":plugin_coordination_service_agent_helper",
        ":plugin_op_kernel",
        ":plugin_variable",
        "//tensorflow/c:c_api_headers",
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_datatype_hdrs",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_hdrs",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:resource_handle_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:logging",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "plugin_op_kernel_helper",
    hdrs = ["plugin_op_kernel_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_plugin_op_kernel",
        ":direct_plugin_op_kernel",
        ":flags",
        ":loose_headers",
        ":plugin_op_kernel",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/flags:flag",
        "@local_xla//xla/tsl/platform:macros",
    ],
)

# For a more maintainable build this target should not exist and the headers
# should  be split into the existing cc_library targets, but this change was
# automatically  done so that we can remove long standing issues and complexity
# in the build system. It's up to the OWNERS of this package to get rid of it or
# not. The use of the textual_hdrs attribute is discouraged, use hdrs instead.
# Here it is used to avoid header parsing errors in packages where the feature
# parse_headers was enabled since loose headers were not being parsed. See
# go/loose-lsc-one-target-approach for more details.
cc_library(
    name = "loose_headers",
    tags = ["avoid_dep"],
    textual_hdrs = ["c_plugin_op_kernel.h"],
    visibility = ["//visibility:private"],
    deps = [
        ":plugin_coordination_service_agent",
        ":plugin_op_kernel",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "plugin_coordination_service_agent",
    hdrs = ["plugin_coordination_service_agent.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "direct_plugin_coordination_service_agent",
    hdrs = ["direct_plugin_coordination_service_agent.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_coordination_service_agent",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/time",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

cc_library(
    name = "c_plugin_coordination_service_agent",
    srcs = ["c_plugin_coordination_service_agent.cc"],
    hdrs = ["c_plugin_coordination_service_agent.h"],
    copts = ["-DTF_CAPI_WEAK"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_coordination_service_agent",
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:tf_buffer",
        "//tensorflow/c:tf_status",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "plugin_coordination_service_agent_helper",
    hdrs = ["plugin_coordination_service_agent_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_plugin_coordination_service_agent",
        ":direct_plugin_coordination_service_agent",
        ":plugin_coordination_service_agent",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/common_runtime/next_pluggable_device:flags",
        "@com_google_absl//absl/flags:flag",
    ],
)

tf_cc_test(
    name = "c_plugin_coordination_service_agent_test",
    srcs = ["c_plugin_coordination_service_agent_test.cc"],
    tags = [
        # TODO(b/317086445): Re-enable windows and mac test.
        "no_mac",
        "no_windows",
    ],
    deps = [
        ":c_plugin_coordination_service_agent",
        "//tensorflow/c:tf_buffer",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/c/experimental/next_pluggable_device:c_api",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/distributed_runtime:call_options",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_client",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
        "@local_xla//xla/tsl/lib/core:status_test_util",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:env_impl",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:test",
        "@local_xla//xla/tsl/platform:test_main",
        "@local_xla//xla/tsl/protobuf:coordination_config_proto_cc",
        "@local_xla//xla/tsl/protobuf:coordination_service_proto_cc",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "plugin_variable",
    hdrs = ["plugin_variable.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "direct_plugin_variable",
    srcs = ["direct_plugin_variable.cc"],
    hdrs = ["direct_plugin_variable.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_variable",
        "//tensorflow/compiler/jit:variable_info",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "c_plugin_variable",
    srcs = ["c_plugin_variable.cc"],
    hdrs = ["c_plugin_variable.h"],
    copts = ["-DTF_CAPI_WEAK"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_variable",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_hdrs",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils.cc",
    ],
    hdrs = [
        "utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "@com_google_absl//absl/log",
        "@local_xla//xla/c:c_api_decl",
    ],
)
