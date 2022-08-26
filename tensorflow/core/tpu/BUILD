# Description: Utilities for TPU Operations

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_libtpu",
    "if_windows",
    "tf_cc_test",
)

package(
    default_visibility = [
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
        "//tensorflow/compiler/tf2xla/kernels:__subpackages__",
        "//tensorflow/compiler/xla:__subpackages__",
        "//tensorflow/compiler/xla/stream_executor/tpu:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
        "//tensorflow/core/profiler/backends/tpu:__subpackages__",
        "//tensorflow/core/tpu:__subpackages__",
        "//tensorflow/dtensor:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "libtftpu_header",
    hdrs = ["libtftpu.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "tpu_embedding_configuration_utils",
    srcs = ["tpu_embedding_configuration_utils.cc"],
    hdrs = ["tpu_embedding_configuration_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "tpu_embedding_errors",
    srcs = ["tpu_embedding_errors.cc"],
    hdrs = ["tpu_embedding_errors.h"],
    deps = [
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "tpu_embedding_errors_test",
    srcs = ["tpu_embedding_errors_test.cc"],
    deps = [
        ":tpu_embedding_errors",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tpu_embedding_optimization_parameters_utils",
    srcs = ["tpu_embedding_optimization_parameters_utils.cc"],
    hdrs = ["tpu_embedding_optimization_parameters_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf/tpu:optimization_parameters_proto_cc",
        "@com_google_absl//absl/base",
    ],
)

cc_library(
    name = "tpu_embedding_output_layout_utils",
    srcs = ["tpu_embedding_output_layout_utils.cc"],
    hdrs = ["tpu_embedding_output_layout_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/framework:tensor_shape_proto_cc",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
    ],
)

cc_library(
    name = "tpu_embedding_configuration_proto_rewrite",
    srcs = ["tpu_embedding_configuration_proto_rewrite.cc"],
    hdrs = ["tpu_embedding_configuration_proto_rewrite.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core/lib/math:math_util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "tpu_embedding_configuration_proto_rewrite_test",
    srcs = ["tpu_embedding_configuration_proto_rewrite_test.cc"],
    deps = [
        ":tpu_embedding_configuration_proto_rewrite",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:test",
        "//tensorflow/core/lib/core:errors",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/platform:casts",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/core/protobuf/tpu:tpu_embedding_configuration_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "tpu_node_device_util",
    srcs = ["tpu_node_device_util.cc"],
    hdrs = ["tpu_node_device_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tpu_compile_interface",
    srcs = ["tpu_compile_interface.cc"],
    hdrs = ["tpu_compile_interface.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tpu_defs",
    srcs = ["tpu_defs.cc"],
    hdrs = ["tpu_defs.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/core:protos_all_cc"],
)

cc_library(
    name = "tpu_configuration",
    srcs = ["tpu_configuration.cc"],
    hdrs = ["tpu_configuration.h"],
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "tpu_init_mode",
    srcs = ["tpu_init_mode.cc"],
    hdrs = ["tpu_init_mode.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "tpu_initializer_helper",
    srcs = ["tpu_initializer_helper.cc"],
    hdrs = ["tpu_initializer_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        ":tpu_api",
        ":tpu_api_dlsym_set_fn",
        ":tpu_executor_init_fns",
        ":tpu_library_init_fns",
        ":tpu_ops_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform/cloud:gcs_file_system",
        "//tensorflow/core/tpu:pjrt_api",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "tpu_api",
    srcs = ["tpu_api.cc"],
    hdrs = ["tpu_api.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        ":tpu_executor_api",
        ":tpu_ops_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_executor_api",
    srcs = ["tpu_executor_api.cc"],
    hdrs = ["tpu_executor_api.h"],
    deps = [
        ":libtftpu_header",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
    ],
)

cc_library(
    name = "pjrt_api",
    srcs = ["pjrt_api.cc"],
    hdrs = ["pjrt_api.h"],
    deps = [
        "//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "tpu_api_dlsym_initializer",
    srcs = if_windows(
        ["tpu_api_dlsym_initializer_windows.cc"],
        otherwise = ["tpu_api_dlsym_initializer.cc"],
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_initializer_helper",
    ],
    # Always link this in, because even if we don't use it directly we want its
    # static initializers to dynamically load API symbols exported from libtpu.so
    alwayslink = True,
)

cc_library(
    name = "tpu_api_dlsym_set_fn",
    hdrs = ["tpu_api_dlsym_set_fn.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "tpu_library_init_fns",
    hdrs = ["tpu_library_init_fns.inc"],
    visibility = ["//visibility:public"],
    deps = [":tpu_executor_init_fns"],
)

cc_library(
    name = "tpu_executor_init_fns",
    hdrs = ["tpu_executor_init_fns.inc"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "virtual_device",
    srcs = ["virtual_device.cc"],
    hdrs = ["virtual_device.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tpu_compile",
    srcs = ["tpu_compile.cc"],
    hdrs = ["tpu_compile.h"],
    deps = [
        ":tpu_defs",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/jit:shape_inference",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:attr_value_proto_cc",
        "//tensorflow/core/framework:node_def_util",
        "//tensorflow/core/framework:versions_proto_cc",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_support",
        "//tensorflow/core/tpu/kernels:tpu_util_hdrs",
    ],
)

cc_library(
    name = "tpu_execute",
    srcs = ["tpu_execute.cc"],
    hdrs = ["tpu_execute.h"],
    deps = [
        ":tpu_api",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_layout",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:maybe_owning_device_memory",
        "//tensorflow/compiler/xla/service:transfer_manager",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
        "//tensorflow/compiler/xla/stream_executor/tpu:status_helper",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_node_context",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_op_executable",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_platform_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_executable_info_proto_cc",
        "//tensorflow/core/tpu/kernels:tpu_execute_op_options",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "tpu_on_demand_compiler",
    srcs = ["tpu_on_demand_compiler.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_cost_analysis",
        "//tensorflow/compiler/xla/service:hlo_module_group",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
        "//tensorflow/compiler/xla/stream_executor/tpu:proto_helper",
        "//tensorflow/compiler/xla/stream_executor/tpu:status_helper",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executable",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_platform_id",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/types:span",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_runtime",
    srcs = [],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_api_dlsym_initializer",
        "//tensorflow/core/tpu:tpu_on_demand_compiler",
        "//tensorflow/core/tpu/ops",
    ],
)

cc_library(
    name = "tpu_ops_c_api_hdrs",
    srcs = [],
    hdrs = [
        "tpu_ops_c_api.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        "//tensorflow/c:tf_tstring",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
        "//tensorflow/compiler/xla/stream_executor/tpu:proto_helper",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_fingerprint_utils",
    srcs = ["tpu_fingerprint_utils.cc"],
    hdrs = ["tpu_fingerprint_utils.h"],
    deps = [
        ":tpu_compile_interface",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core/lib/core:status",
        "//tensorflow/core/lib/strings:proto_serialization",
    ],
)

cc_library(
    name = "tpu_model_server_initializer",
    srcs = ["tpu_model_server_initializer.cc"],
    hdrs = ["tpu_model_server_initializer.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":libtftpu_header",
        ":tpu_api",
        ":tpu_api_dlsym_set_fn",
        ":tpu_executor_init_fns",
        ":tpu_initializer_helper",
        ":tpu_library_init_fns",
        ":tpu_ops_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_c_api_hdrs",
        "//tensorflow/core:lib",
    ],
    alwayslink = True,
)

cc_library(
    name = "tpu_global_init",
    srcs = ["tpu_global_init.cc"],
    hdrs = ["tpu_global_init.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu_defs",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:tpu_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/core/tpu/graph_rewrite:distributed_tpu_configuration_rewrite_pass",
        "//tensorflow/core/tpu/graph_rewrite:distributed_tpu_rewrite_helpers",
    ] + if_libtpu(["//tensorflow/compiler/jit"]),
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
    textual_hdrs = ["tpu_global_init.h"],
    visibility = ["//tensorflow_serving/model_servers:__pkg__"],
)
