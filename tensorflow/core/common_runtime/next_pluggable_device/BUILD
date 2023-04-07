load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
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
        "//tensorflow/compiler/jit:pjrt_device_context",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tfrt/common:pjrt_state",
        "@com_google_absl//absl/flags:flag",
    ],
)

cc_library(
    name = "next_pluggable_device_api",
    srcs = ["next_pluggable_device_api.cc"],
    hdrs = ["next_pluggable_device_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
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
        ":pjrt_compile_on_demand_op",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime/next_pluggable_device/c:plugin_c_api_hdrs",
        "//tensorflow/tsl/platform:errors",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "pjrt_compile_on_demand_op",
    srcs = [
        "pjrt_compile_on_demand_op.cc",
    ],
    hdrs = [
        "pjrt_compile_on_demand_op.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":next_pluggable_device",
        ":next_pluggable_device_api",
        ":utils",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/jit:device_compiler_client",
        "//tensorflow/compiler/jit:variable_info",
        "//tensorflow/compiler/jit:variable_info_util",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/jit:xla_launch_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "//tensorflow/core/util:determinism",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:statusor",
    ],
    alwayslink = 1,
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
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:bfloat16",
        "//tensorflow/core/platform:status",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/platform:status",
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
    name = "c_plugin_op_kernel",
    srcs = ["c_plugin_op_kernel.cc"],
    hdrs = ["c_plugin_op_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":c_plugin_variable",
        ":plugin_coordination_service_agent",
        ":plugin_coordination_service_agent_helper",
        ":plugin_op_kernel",
        ":plugin_variable",
        "//tensorflow/c:c_api_internal",
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_buffer",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:resource_handle_proto_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "plugin_op_kernel_helper",
    hdrs = ["plugin_op_kernel_helper.h"],
    defines = select({
        "tf_npd_use_c_api_enabled": ["TF_NEXT_PLUGGABLE_DEVICE_USE_C_API"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":loose_headers",
        ":plugin_op_kernel",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_status_helper",
    ] + select({
        "tf_npd_use_c_api_enabled": [":c_plugin_op_kernel"],
        "//conditions:default": [
            ":direct_plugin_op_kernel",
        ],
    }),
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
    visibility = [":__pkg__"],
)

cc_library(
    name = "plugin_coordination_service_agent",
    hdrs = ["plugin_coordination_service_agent.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core/platform:statusor"],
)

cc_library(
    name = "direct_plugin_coordination_service_agent",
    hdrs = ["direct_plugin_coordination_service_agent.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_coordination_service_agent",
        "//tensorflow/core/platform:status",
        "//tensorflow/tsl/distributed_runtime/coordination:coordination_service_agent",
    ],
)

cc_library(
    name = "c_plugin_coordination_service_agent",
    srcs = ["c_plugin_coordination_service_agent.cc"],
    hdrs = ["c_plugin_coordination_service_agent.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_coordination_service_agent",
        "//tensorflow/c:kernels_experimental_hdrs",
        "//tensorflow/c:tf_buffer_internal",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "plugin_coordination_service_agent_helper",
    hdrs = ["plugin_coordination_service_agent_helper.h"],
    defines = select({
        "tf_npd_use_c_api_enabled": ["TF_NEXT_PLUGGABLE_DEVICE_USE_C_API"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_coordination_service_agent",
        "//tensorflow/c:kernels_hdrs",
        "//tensorflow/c:tf_status_helper",
    ] + select({
        "tf_npd_use_c_api_enabled": [
            ":c_plugin_coordination_service_agent",
        ],
        "//conditions:default": [
            ":direct_plugin_coordination_service_agent",
        ],
    }),
)

cc_library(
    name = "plugin_variable",
    hdrs = ["plugin_variable.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:status",
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
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "c_plugin_variable",
    srcs = ["c_plugin_variable.cc"],
    hdrs = ["c_plugin_variable.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_variable",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/experimental/next_pluggable_device:c_api_hdrs",
        "//tensorflow/core:framework",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:status",
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
        "//tensorflow/compiler/xla/c:c_api_decl",
    ],
)
