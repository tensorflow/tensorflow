load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_is_configured_compat",
    "tf_copts",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cc_test_gpu")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_cupti_utils_cuda_deps",
    "tf_additional_device_tracer_srcs",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_cuda_library(
    name = "device_tracer",
    srcs = tf_additional_device_tracer_srcs(),
    copts = tf_copts(),
    cuda_deps = [
        "//tensorflow/core/profiler/internal/gpu:cupti_tracer",
        "//tensorflow/core/profiler/internal/gpu:cupti_wrapper",
    ],
    deps = [
        ":cupti_utils",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/internal:annotation_stack",
        "//tensorflow/core/profiler/internal:parse_annotation",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = 1,
)

tf_cc_test_gpu(
    name = "device_tracer_test",
    size = "small",
    srcs = ["device_tracer_test.cc"],
    args = ["--heap_check=local"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + ["nomac"],
    deps = [
        ":device_tracer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:direct_session_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/profiler/utils:xplane_visitor",
    ],
)

tf_cuda_library(
    name = "cupti_interface",
    hdrs = if_cuda_is_configured_compat(["cupti_interface.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:platform_base",
        "//tensorflow/stream_executor/cuda:cupti_stub",
    ],
)

tf_cuda_library(
    name = "cupti_wrapper",
    srcs = if_cuda_is_configured_compat(["cupti_wrapper.cc"]),
    hdrs = if_cuda_is_configured_compat(["cupti_wrapper.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//tensorflow/stream_executor/cuda:cupti_stub",
    ],
)

tf_cuda_library(
    name = "cupti_tracer",
    srcs = if_cuda_is_configured_compat(["cupti_tracer.cc"]),
    hdrs = if_cuda_is_configured_compat(["cupti_tracer.h"]),
    copts = tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        ":cupti_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/internal:annotation_stack",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cuda_library(
    name = "cupti_utils",
    srcs = if_cuda_is_configured_compat(["cupti_utils.cc"]),
    copts = tf_copts(),
    cuda_deps = [
        ":cupti_interface",
        ":cupti_wrapper",
    ] + tf_additional_cupti_utils_cuda_deps(),
    visibility = ["//visibility:public"],
)
