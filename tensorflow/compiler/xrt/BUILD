# Description: Operations defined for XRT

load(
    "//tensorflow:tensorflow.bzl",
    "tf_custom_op_py_library",
    "tf_gen_op_libs",
    "tf_gen_op_wrapper_py",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)

package(
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_proto_library_cc(
    name = "xrt_proto",
    srcs = ["xrt.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto",
        "//tensorflow/compiler/xla:xla_data_proto",
        "//tensorflow/compiler/xla:xla_proto",
        "//tensorflow/compiler/xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

tf_proto_library_py(
    name = "xrt_proto",  # bzl adds a _py suffix
    srcs = ["xrt.proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "xrt_utils",
    srcs = [
        "xrt_compilation_cache.cc",
        "xrt_device.cc",
        "xrt_memory_manager.cc",
        "xrt_metrics.cc",
        "xrt_state.cc",
        "xrt_util.cc",
    ],
    hdrs = [
        "xrt_compilation_cache.h",
        "xrt_device.h",
        "xrt_memory_manager.h",
        "xrt_metrics.h",
        "xrt_refptr.h",
        "xrt_state.h",
        "xrt_util.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":xrt_proto_cc",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:backend",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:shaped_buffer",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor:device_memory_allocator",
        "@com_google_absl//absl/memory",
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

tf_gen_op_wrapper_py(
    name = "xrt_ops_wrapper_py",
    out = "xrt_ops.py",
    deps = [
        ":xrt_compile_ops_op_lib",
        ":xrt_execute_op_op_lib",
        ":xrt_state_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "xrt_ops",
    kernels = ["//tensorflow/compiler/xrt/kernels:xrt_ops"],
    visibility = ["//visibility:public"],
    deps = [
        ":xrt_ops_wrapper_py",
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
