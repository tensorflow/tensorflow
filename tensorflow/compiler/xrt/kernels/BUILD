load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/tf2xla:friends",
    ],
)

WITH_TPU_SUPPORT = "//tensorflow:with_tpu_support"

DEFAULT = "//conditions:default"

cc_library(
    name = "xrt_state_ops",
    hdrs = ["xrt_state_ops.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xrt_tpu_ops",
    srcs = [
        "tpu_compile_ops.cc",
        "tpu_execute_op.cc",
        "tpu_state_op.cc",
    ],
    visibility = [":friends"],
    deps = [
        ":xrt_state_ops",
        "//tensorflow/compiler/jit:xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:compile_only_client",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:dump",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_tpu_utils",
        "//tensorflow/compiler/xrt:xrt_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tpu:tpu_api",
        "//tensorflow/core/tpu:tpu_configuration",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_execute",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_entry",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_interface",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_key",
        "//tensorflow/core/tpu/kernels:tpu_compilation_cache_lookup",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_common",
        "//tensorflow/core/tpu/kernels:tpu_compile_op_hdrs",
        "//tensorflow/core/tpu/kernels:tpu_mesh_state_interface",
        "//tensorflow/core/tpu/kernels:tpu_op_consts",
        "//tensorflow/core/tpu/kernels:tpu_op_util",
        "//tensorflow/core/tpu/kernels:tpu_program_group",
        "//tensorflow/core/tpu/kernels:tpu_program_group_interface",
        "//tensorflow/stream_executor:stream_executor_headers",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xrt_ops",
    srcs = [
        "xrt_compile_ops.cc",
        "xrt_execute_op.cc",
        "xrt_state_ops.cc",
    ],
    visibility = [":friends"],
    deps = select({
        WITH_TPU_SUPPORT: [":xrt_tpu_ops"],
        DEFAULT: [],
    }) + [
        ":xrt_state_ops",
        "@com_google_absl//absl/strings",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/client:xla_computation",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xrt:xrt_compile_ops_op_lib",
        "//tensorflow/compiler/xrt:xrt_execute_op_op_lib",
        "//tensorflow/compiler/xrt:xrt_proto_cc",
        "//tensorflow/compiler/xrt:xrt_state_ops_op_lib",
        "//tensorflow/compiler/xrt:xrt_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor:stream_executor_headers",
    ],
    alwayslink = 1,
)
