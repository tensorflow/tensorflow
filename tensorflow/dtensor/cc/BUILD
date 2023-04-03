# DTensor C++ runtime and libraries.

load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_dtensor_tpu_dependencies",
)
load("//tensorflow:tensorflow.default.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "if_google", "if_libtpu")

default_visibility = [
    "//tensorflow/dtensor:dtensor-internal",
    "//tensorflow:__pkg__",
]

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = default_visibility,
    licenses = ["notice"],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
)

cc_library(
    name = "dstatus",
    hdrs = ["dstatus.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "dtensor_utils",
    srcs = ["dtensor_utils.cc"],
    hdrs = ["dtensor_utils.h"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "tensor_layout",
    srcs = ["tensor_layout.cc"],
    hdrs = ["tensor_layout.h"],
    deps = [
        ":dstatus",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "small_constant_optimization",
    srcs = ["small_constant_optimization.cc"],
    hdrs = ["small_constant_optimization.h"],
    deps = [
        ":constants",
        ":tensor_layout",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "tensor_with_layout",
    srcs = ["tensor_with_layout.cc"],
    hdrs = ["tensor_with_layout.h"],
    deps = [
        ":constants",
        ":tensor_layout",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:fingerprint",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "dtensor_device_util",
    srcs = ["dtensor_device_util.cc"],
    hdrs = ["dtensor_device_util.h"],
    deps = [
        ":constants",
        ":dstatus",
        ":small_constant_optimization",
        ":tensor_layout",
        ":tensor_with_layout",
        "//tensorflow/c:safe_ptr",
        "//tensorflow/c:tf_datatype",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/c/eager/parallel_device:parallel_device_lib",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/tsl/platform:refcount",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "dtensor_ops",
    srcs = [
        "dtensor_ops.cc",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

# These ops are created only by DTensor MLIR passes and never by users, so they don't need Python wrappers.
cc_library(
    name = "dtensor_meta_ops",
    srcs = [
        "dtensor_meta_ops.cc",
    ],
    deps = [
        ":tensor_layout",
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_system_interface",
    srcs = ["tpu_system_interface.cc"],
    hdrs = ["tpu_system_interface.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "save_restore_util",
    srcs = ["save_restore_util.cc"],
    hdrs = ["save_restore_util.h"],
    deps = [
        ":dstatus",
        ":tensor_layout",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/dtensor/mlir:value_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "dtensor_tpu_ops",
    srcs = ["dtensor_tpu_ops.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "dtensor_tpu_kernels",
    srcs = [
        "dtensor_tpu_kernels.cc",
    ],
    tags = [
        "no_rocm",
        "tpu",
    ],  # Disable building of TPU kernels on non-TPU platforms.
    deps = [
        ":dstatus",
        ":tpu_system_interface",
        "//tensorflow/c:tf_status_headers",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_api",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_executor_base",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_ops_c_api_hdrs",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_topology_external",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_interface",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_local_lookup",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_lookup",
        "//tensorflow/core/tpu/kernels:tpu_configuration_ops",
        "//tensorflow/core/tpu/kernels:tpu_embedding_engine_state_interface",
        "//tensorflow/core/tpu/kernels:tpu_mesh_state_interface",
        "//tensorflow/core/tpu/kernels:tpu_op_consts",
        "//tensorflow/core/tpu/kernels:tpu_pod_state",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dtensor_graph_to_mlir_pass",
    srcs = ["dtensor_graph_to_mlir_pass.cc"],
    hdrs = ["dtensor_graph_to_mlir_pass.h"],
    deps = [
        ":constants",
        ":dtensor_utils",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/dtensor/mlir:dtensor_mlir_passes",
        "//tensorflow/dtensor/mlir:tf_dtensor_dialect",
        "//tensorflow/dtensor/mlir/dtensor_dialect:Dialect",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "default_parallel_executor_lib",
    deps = if_libtpu(
        if_false = if_google(
            google_value = ["//tensorflow/dtensor/cc/google:default_parallel_executor"],
            oss_value = [":default_parallel_executor"],
        ),
        if_true = [":default_parallel_executor"],
    ),
)

cc_library(
    name = "default_parallel_executor",
    srcs = ["default_parallel_executor.cc"],
    deps = [
        ":parallel_executor_interface",
        "//tensorflow/core/platform:logging",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "parallel_executor_interface",
    hdrs = ["parallel_executor.h"],
    deps = [
        ":tensor_layout",
        ":tensor_with_layout",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/compiler/xla/pjrt:pjrt_future",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "dtensor_device_cc",
    srcs = ["dtensor_device.cc"],
    hdrs = ["dtensor_device.h"],
    deps = [
        ":constants",
        ":default_parallel_executor_lib",
        ":dstatus",
        ":dtensor_device_util",
        ":dtensor_graph_to_mlir_pass",
        ":dtensor_meta_ops",
        ":dtensor_ops",
        ":dtensor_tpu_ops",
        ":parallel_executor_interface",
        ":small_constant_optimization",
        ":tensor_layout",
        ":tpu_system_interface",
        "//tensorflow/c:c_api",
        "//tensorflow/c:c_api_experimental",
        "//tensorflow/c:tf_status_helper",
        "//tensorflow/c:tf_tensor_internal",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_experimental",
        "//tensorflow/c/eager:tfe_context_internal",
        "//tensorflow/c/eager:tfe_op_attrs_internal",
        "//tensorflow/c/eager:tfe_tensorhandle_internal",
        "//tensorflow/c/eager/parallel_device:parallel_device_lib",
        "//tensorflow/compiler/mlir/tensorflow:export_graphdef",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_decl",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_platform_interface",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_topology_external",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:device_set",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/common_runtime/eager:eager_operation",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/dtensor/mlir:layout_parsing",
        "//tensorflow/dtensor/proto:layout_proto_cc",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/util:env_var",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ] + tf_dtensor_tpu_dependencies(),
)

cc_library(
    name = "layout_to_xla_sharding",
    srcs = ["xla_spmd/layout_to_xla_sharding.cc"],
    hdrs = ["xla_spmd/layout_to_xla_sharding.h"],
    deps = [
        ":dstatus",
        ":tensor_layout",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "mesh_type",
    hdrs = ["mesh_type.h"],
    deps = [
        ":tensor_layout",
        "//tensorflow/c:conversion_macros",
    ],
)
