load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "tf_cuda_library",
)
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_additional_device_tracer_srcs",
    "tf_kernel_tests_linkstatic",
)
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load(
    "//tensorflow/stream_executor:build_defs.bzl",
    "tf_additional_cupti_deps",
)
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")
load(
    "//tensorflow/core/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)

package(
    default_visibility = ["//tensorflow:internal"],
    features = [
        "-layering_check",
    ],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "device_tracer",
    srcs = tf_additional_device_tracer_srcs(),
    copts = tf_profiler_copts() + tf_copts(),
    cuda_deps = [
        "//tensorflow/core/profiler/internal/gpu:cupti_tracer",
        "//tensorflow/core/profiler/internal/gpu:cupti_wrapper",
        "//tensorflow/core/profiler/internal/gpu:rocm_tracer",
    ],
    deps = [
        ":cupti_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = 1,
)

tf_cuda_cc_test(
    name = "device_tracer_test",
    size = "small",
    srcs = ["device_tracer_test.cc"],
    args = ["--heap_check="],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + [
        "nomac",
        "gpu_cupti",
    ],
    deps = [
        ":device_tracer",
        "@com_google_absl//absl/strings",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ] + if_cuda_is_configured([
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cupti_headers",
    ]),
)

tf_cuda_library(
    name = "cupti_interface",
    hdrs = if_cuda(["cupti_interface.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:platform_base",
    ] + tf_additional_cupti_deps(),
)

tf_cuda_library(
    name = "mock_cupti",
    testonly = 1,
    hdrs = if_cuda(["mock_cupti.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    cuda_deps = [
        ":cupti_interface",
    ],
    deps = [
        "//tensorflow/core:test",
    ],
)

tf_cuda_library(
    name = "cupti_error_manager",
    srcs = if_cuda(["cupti_error_manager.cc"]),
    hdrs = if_cuda(["cupti_error_manager.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    cuda_deps = [
        ":cupti_interface",
        ":cupti_wrapper",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cuda_cc_test(
    name = "cupti_error_manager_test",
    size = "small",
    srcs = ["cupti_error_manager_test.cc"],
    linkstatic = tf_kernel_tests_linkstatic(),
    tags = tf_cuda_tests_tags() + [
        "nomac",
        "gpu_cupti",
    ],
    deps = [
        "//tensorflow/core:test_main",
    ] + if_cuda_is_configured([
        ":cuda_test",
        ":cupti_error_manager",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":mock_cupti",
        "@com_google_absl//absl/memory",
        "//tensorflow/core/profiler/utils:time_utils",
    ]),
)

cuda_library(
    name = "cuda_test",
    testonly = 1,
    srcs = ["cuda_test.cu.cc"],
    hdrs = ["cuda_test.h"],
    copts = select({
        "@local_config_cuda//cuda:using_nvcc": [
            "-nvcc_options",
            "ptxas-options=-v",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:test",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

# Rationale for linkstatic: The symbols in libcupti_static.a have hidden
# visibility. The wrapper will fail to find them if it's ever built as a
# shared library. This is the same issue as b/11094727. Always linking
# the wrapper statically works around the issue. An alternative would be
# to patch libcupti_static, but it's not worth the trouble considering
# that the wrapper is about the only direct user.
tf_cuda_library(
    name = "cupti_wrapper",
    srcs = if_cuda(["cupti_wrapper.cc"]),
    hdrs = if_cuda(["cupti_wrapper.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
    ] + tf_additional_cupti_deps(),
)

tf_cuda_library(
    name = "cupti_tracer",
    srcs = if_cuda(["cupti_tracer.cc"]),
    hdrs = if_cuda(["cupti_tracer.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_utils",
        ":nvtx_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/internal/cpu:annotation_stack",
        "//tensorflow/core/profiler/lib:scoped_annotation",
        "//tensorflow/core/profiler/utils:buffer_pool",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cuda_library(
    name = "rocm_tracer",
    srcs = if_rocm(["rocm_tracer.cc"]),
    hdrs = if_rocm(["rocm_tracer.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/internal/cpu:annotation_stack",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/stream_executor/rocm:roctracer_wrapper",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cuda_library(
    name = "nvtx_utils",
    srcs = if_cuda(["nvtx_utils.cc"]),
    hdrs = if_cuda(["nvtx_utils.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_cuda_library(
    name = "cupti_collector",
    srcs = if_cuda(["cupti_collector.cc"]),
    hdrs = if_cuda(["cupti_collector.h"]),
    copts = tf_profiler_copts() + tf_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:parse_annotation",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ] + tf_additional_cupti_deps(),
)

cc_library(
    name = "cupti_collector_header",
    hdrs = ["cupti_collector.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:parse_annotation",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_library(
    name = "cupti_utils",
    srcs = if_cuda(["cupti_utils.cc"]),
    copts = tf_profiler_copts() + tf_copts(),
    cuda_deps = [
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_wrapper",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
