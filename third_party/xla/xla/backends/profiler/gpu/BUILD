load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library", "if_cuda")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
)

cc_library(
    name = "device_tracer",
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    deprecation = "Use :device_tracer_{cuda,rocm} instead",
    deps = if_cuda([":device_tracer_cuda"]) + if_rocm([":device_tracer_rocm"]),
    alwayslink = 1,
)

cc_library(
    name = "device_tracer_cuda",
    srcs = ["device_tracer_cuda.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "manual",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_tracer",
        ":cupti_utils",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "device_tracer_rocm",
    srcs = ["device_tracer_rocm.cc"],
    tags = [
        "gpu",
        "manual",
        "rocm-only",
    ],
    deps = [
        ":rocm_collector",
        ":rocm_tracer",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:abi",
        "@local_tsl//tsl/platform:env_time",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:thread_annotations",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
    ],
)

cc_library(
    name = "cupti_interface",
    hdrs = ["cupti_interface.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/cuda:cupti",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "mock_cupti",
    testonly = 1,
    hdrs = ["mock_cupti.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_interface",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cupti_error_manager",
    srcs = ["cupti_error_manager.cc"],
    hdrs = ["cupti_error_manager.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        ":cupti_wrapper",
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
    tags = [
        "cuda-only",
        "no_mac",
    ],
    deps = [
        ":cuda_test",
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":mock_cupti",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:test",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cuda_library(
    name = "cuda_test",
    testonly = 1,
    srcs = ["cuda_test.cu.cc"],
    hdrs = ["cuda_test.h"],
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
cc_library(
    name = "cupti_wrapper",
    srcs = [
        "cupti_wrapper.cc",
        "cupti_wrapper_stub.cc",
    ],
    hdrs = ["cupti_wrapper.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    linkstatic = 1,
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_interface",
        "//xla/tsl/cuda:cupti",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cc_library(
    name = "cupti_tracer",
    srcs = ["cupti_tracer.cc"],
    hdrs = ["cupti_tracer.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
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
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "cupti_profiler",
    srcs = ["cupti_profiler.cc"],
    hdrs = ["cupti_profiler.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        ":cupti_utils",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:types",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "rocm_collector",
    srcs = ["rocm_collector.cc"],
    hdrs = ["rocm_collector.h"],
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

cc_library(
    name = "rocm_tracer",
    srcs = ["rocm_tracer.cc"],
    hdrs = ["rocm_tracer.h"],
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
        "@local_config_rocm//rocm:rocm_headers",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:macros",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "nvtx_utils",
    srcs = ["nvtx_utils.cc"],
    hdrs = ["nvtx_utils.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform",
        "@local_tsl//tsl/platform:macros",
    ],
)

cc_library(
    name = "cupti_collector",
    srcs = ["cupti_collector.cc"],
    hdrs = ["cupti_collector.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_buffer_events",
        ":cupti_interface",
        "//xla/tsl/cuda",
        "//xla/tsl/cuda:cupti",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:abi",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "cupti_buffer_events",
    srcs = ["cupti_buffer_events.cc"],
    hdrs = ["cupti_buffer_events.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        ":cupti_utils",
        ":kernel_details",
        "//xla/tsl/cuda:cupti",
        "//xla/tsl/profiler/utils:buffer_pool",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:mutex",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "cupti_utils",
    srcs = ["cupti_utils.cc"],
    hdrs = ["cupti_utils.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_wrapper",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/memory",
        "@local_tsl//tsl/platform:logging",
        "@local_tsl//tsl/platform:stringpiece",
    ],
    alwayslink = 1,
)

xla_cc_test(
    name = "cupti_buffer_events_test",
    size = "small",
    srcs = ["cupti_buffer_events_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
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

cc_library(
    name = "kernel_details",
    hdrs = ["kernel_details.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:non_prod"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)
