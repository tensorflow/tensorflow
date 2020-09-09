package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/tf2xla:friends",
    ],
)

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
    name = "xrt_ops",
    srcs = [
        "xrt_compile_ops.cc",
        "xrt_execute_op.cc",
        "xrt_state_ops.cc",
    ],
    visibility = [":friends"],
    deps = [
        ":xrt_state_ops",
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
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)
