# Description: Operations defined for XRT

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
)

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_libs",
)
load("//tensorflow/compiler/xla:xla.bzl", "xla_proto_library")

xla_proto_library(
    name = "xrt_proto",
    srcs = ["xrt.proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
)

cc_library(
    name = "xrt_utils",
    srcs = [
        "xrt_compilation_cache.cc",
        "xrt_device.cc",
        "xrt_state.cc",
        "xrt_util.cc",
    ],
    hdrs = [
        "xrt_compilation_cache.h",
        "xrt_device.h",
        "xrt_state.h",
        "xrt_util.h",
    ],
    deps = [
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:device_memory_allocator",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/stream_executor",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "xrt_compile_ops",
        "xrt_state_ops",
        "xrt_execute_op",
    ],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "xrt_server",
    visibility = ["//visibility:public"],
    deps = [
        ":xrt_compile_ops_op_lib",
        ":xrt_execute_op_op_lib",
        ":xrt_state_ops_op_lib",
        "//tensorflow/compiler/xrt/kernels:xrt_ops",
    ],
)
