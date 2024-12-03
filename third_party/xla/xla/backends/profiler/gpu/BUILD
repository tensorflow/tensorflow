load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "if_nvcc",
    "internal_visibility",
    "tsl_copts",
    "tsl_gpu_library",
)
load(
    "//xla/tsl/platform:build_config.bzl",
    "tf_additional_device_tracer_srcs",
)
load(
    "//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load("//xla/tsl/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
)

tsl_gpu_library(
    name = "device_tracer",
    srcs = tf_additional_device_tracer_srcs(),
    copts = tf_profiler_copts() + tsl_copts(),
    deps = [
        ":cupti_utils",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ] + if_cuda([
        # keep sorted
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_tracer",
        ":cupti_wrapper",
    ]) + if_rocm([
        # keep sorted
        ":rocm_collector",
        ":rocm_tracer",
    ]),
    alwayslink = 1,
)

tsl_gpu_library(
    name = "cupti_interface",
    hdrs = if_cuda(["cupti_interface.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ] + if_cuda(["//xla/tsl/cuda:cupti"]),
)

tsl_gpu_library(
    name = "mock_cupti",
    testonly = 1,
    hdrs = if_cuda(["mock_cupti.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_interface",
    ],
    deps = [
        "@local_tsl//tsl/platform:test",
    ],
)

tsl_gpu_library(
    name = "cupti_error_manager",
    srcs = if_cuda(["cupti_error_manager.cc"]),
    hdrs = if_cuda(["cupti_error_manager.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_interface",
        ":cupti_wrapper",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/debugging:leak_check",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

xla_test(
    name = "cupti_error_manager_test",
    size = "small",
    srcs = ["cupti_error_manager_test.cc"],
    backends = ["gpu"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "no_mac",
    ],
    deps = [
        ":cupti_interface",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ] + if_cuda_is_configured([
        ":cuda_test",
        ":cupti_error_manager",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":mock_cupti",
        "@com_google_absl//absl/memory",
        "//xla/tsl/profiler/utils:time_utils",
    ]),
)

cuda_library(
    name = "cuda_test",
    testonly = 1,
    srcs = ["cuda_test.cu.cc"],
    hdrs = ["cuda_test.h"],
    copts = if_nvcc([
        "-nvcc_options",
        "ptxas-options=-v",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@local_tsl//tsl/platform:test",
    ],
)

# Rationale for linkstatic: The symbols in libcupti_static.a have hidden
# visibility. The wrapper will fail to find them if it's ever built as a
# shared library. This is the same issue as b/11094727. Always linking
# the wrapper statically works around the issue. An alternative would be
# to patch libcupti_static, but it's not worth the trouble considering
# that the wrapper is about the only direct user.
tsl_gpu_library(
    name = "cupti_wrapper",
    srcs = if_cuda([
        "cupti_wrapper.cc",
        "cupti_wrapper_stub.cc",
    ]),
    hdrs = if_cuda(["cupti_wrapper.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
    ] + if_cuda(["//xla/tsl/cuda:cupti"]),
)

tsl_gpu_library(
    name = "cupti_tracer",
    srcs = if_cuda(["cupti_tracer.cc"]),
    hdrs = if_cuda(["cupti_tracer.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_utils",
        ":nvtx_utils",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:per_thread",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:types",
    ],
)

tsl_gpu_library(
    name = "cupti_profiler",
    srcs = if_cuda(["cupti_profiler.cc"]),
    hdrs = if_cuda(["cupti_profiler.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

tsl_gpu_library(
    name = "rocm_collector",
    srcs = if_rocm(["rocm_collector.cc"]),
    hdrs = if_rocm(["rocm_collector.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:abi",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
    ],
)

tsl_gpu_library(
    name = "rocm_tracer",
    srcs = if_rocm(["rocm_tracer.cc"]),
    hdrs = if_rocm(["rocm_tracer.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # TODO(b/360374983): Remove this tag once the target can be built without --config=rocm.
        "manual",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        ":rocm_collector",
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
    ],
)

tsl_gpu_library(
    name = "nvtx_utils",
    srcs = if_cuda(["nvtx_utils.cc"]),
    hdrs = if_cuda(["nvtx_utils.h"]),
    copts = tf_profiler_copts() + tsl_copts(),
    deps = [
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:macros",
    ],
)

tsl_gpu_library(
    name = "cupti_collector",
    srcs = if_cuda(["cupti_collector.cc"]),
    hdrs = ["cupti_collector.h"],
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_buffer_events",
        ":cupti_interface",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:abi",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ] + if_cuda([
        "//xla/tsl/cuda:cupti",
        "//xla/tsl/cuda",
    ]),
)

tsl_gpu_library(
    name = "cupti_buffer_events",
    srcs = if_cuda(["cupti_buffer_events.cc"]),
    hdrs = ["cupti_buffer_events.h"],
    copts = tf_profiler_copts() + tsl_copts(),
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        "//xla/tsl/profiler/utils:buffer_pool",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
    ] + if_cuda(["//xla/tsl/cuda:cupti"]),
)

tsl_gpu_library(
    name = "cupti_utils",
    srcs = if_cuda(["cupti_utils.cc"]),
    copts = tf_profiler_copts() + tsl_copts(),
    cuda_deps = [
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_wrapper",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:stringpiece",
        "//xla/tsl/util:env_var",
    ],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)

xla_test(
    name = "cupti_buffer_events_test",
    size = "small",
    srcs = ["cupti_buffer_events_test.cc"],
    backends = ["cpu"],
    copts = tf_profiler_copts() + tsl_copts(),
    tags = [
        "no_mac",
    ],
    deps = [
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_tracer",
        ":cupti_utils",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)
