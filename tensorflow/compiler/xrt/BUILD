# Description: Operations defined for XRT

# Placeholder: load py_proto_library
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
)
load("//tensorflow:tensorflow.default.bzl", "tf_custom_op_py_strict_library", "tf_gen_op_libs")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//learning/brain:__subpackages__",
        "//tensorflow/compiler/xrt:__subpackages__",
    ],
    licenses = ["notice"],
)

tf_proto_library(
    name = "xrt_proto",
    srcs = ["xrt.proto"],
    cc_api_version = 2,
    protodeps = [
        "//tensorflow/compiler/tf2xla:host_compute_metadata_proto",
        "@local_xla//xla:xla_data_proto",
        "@local_xla//xla:xla_proto",
        "@local_xla//xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "xrt_tpu_utils",
    srcs = [
        "xrt_tpu_device.cc",
    ],
    hdrs = [
        "xrt_tpu_device.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/tpu:tpu_configuration",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
    ],
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
    copts = if_cuda(["-DGOOGLE_CUDA=1"]),
    visibility = ["//visibility:public"],
    deps = [
        ":xrt_proto_cc",
        "//tensorflow/compiler/jit:xla_device",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/common_runtime/gpu:gpu_runtime",
        "//tensorflow/core/platform:regexp",
        "//tensorflow/core/profiler/lib:traceme",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/synchronization",
        "@local_xla//xla:debug_options_flags",
        "@local_xla//xla:literal",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla:types",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla:xla_proto_cc",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:backend",
        "@local_xla//xla/service:executable",
        "@local_xla//xla/service:shaped_buffer",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
    ],
)

tf_gen_op_libs(
    op_lib_names = [
        "xrt_compile_ops",
        "xrt_state_ops",
        "xrt_execute_op",
    ],
    deps = [
        "//tensorflow/compiler/jit:common",
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "xrt_ops_wrapper_py",
    out = "xrt_ops.py",
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [
        ":xrt_compile_ops_op_lib",
        ":xrt_execute_op_op_lib",
        ":xrt_state_ops_op_lib",
    ],
)

tf_custom_op_py_strict_library(
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

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "xrt_proto_py_pb2",
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":xrt_proto"],
# )
# copybara:uncomment_end
