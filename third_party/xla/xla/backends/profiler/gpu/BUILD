load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tests:build_defs.bzl", "xla_test")
load(
    "//xla/tsl:tsl.bzl",
    "if_google",
    "internal_visibility",
)
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured", "if_cuda_newer_than")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility(["//xla:internal"]),
)

cc_library(
    name = "device_tracer",
    visibility = ["//visibility:public"],
    deps =
        if_cuda_is_configured([
            ":device_tracer_cuda",
        ]) + if_rocm_is_configured([
            ":device_tracer_rocm",
        ]),
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
        ":cupti_tracer_options_utils",
        ":gpu_metadata",
        "//xla/tsl/platform:errors",
        "//xla/tsl/profiler/utils:time_utils",
        "//xla/tsl/util:env_var",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
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
        ":rocm_tracer_utils",
        "//xla:debug_options_flags",
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/platform:env_time",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@tsl//tsl/profiler/lib:profiler_factory",
        "@tsl//tsl/profiler/lib:profiler_interface",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "cupti_interface",
    hdrs = ["cupti_interface.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//xla/tsl/cuda:cupti",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:types",
    ],
)

cc_library(
    name = "cuda_version_variants",
    srcs = [
        "cuda_version_variants.cc",
    ] + if_cuda_newer_than(
        "12_0",
        ["cuda_version_12000_newer.cc"],
        ["cuda_version_12000_older.cc"],
    ) + if_cuda_newer_than(
        "12_8",
        ["cuda_version_12080_newer.cc"],
        ["cuda_version_12080_older.cc"],
    ),
    hdrs = ["cuda_version_variants.h"],
    tags = [
        "cuda-only",
        "gpu",
        "requires-gpu-nvidia",
    ],
    deps = [
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

xla_cc_test(
    name = "cuda_version_variants_test",
    srcs = ["cuda_version_variants_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "no_mac",
        "nomsan",  # Disable  msan for cuda related tests.
        "requires-gpu-nvidia",
    ],
    deps = [
        ":cuda_version_variants",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@tsl//tsl/platform:status_matchers",
        "@tsl//tsl/platform:test",
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
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "cupti_error_manager",
    srcs = ["cupti_error_manager.cc"],
    hdrs = ["cupti_error_manager.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_interface",
        ":cupti_wrapper",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:thread_annotations",
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
        ":cupti_collector",
        ":cupti_error_manager",
        ":cupti_interface",
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":mock_cupti",
        "//xla/tsl/profiler/utils:time_utils",
        "@com_google_googletest//:gtest_main",
    ],
)

cuda_library(
    name = "cuda_test",
    testonly = 1,
    srcs = ["cuda_test.cu.cc"],
    hdrs = ["cuda_test.h"],
    tags = ["cuda-only"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_googletest//:gtest_for_library",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cuda_runtime",
        "@tsl//tsl/profiler/lib:scoped_annotation",
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
    name = "cupti_tracer_options_utils",
    srcs = ["cupti_tracer_options_utils.cc"],
    hdrs = ["cupti_tracer_options_utils.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_tracer",
        "//xla/tsl/platform:errors",
        "//xla/tsl/profiler/utils:profiler_options_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        ":cuda_version_variants",
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_marker_data_parser",
        ":cupti_pm_sampler_factory",
        ":cupti_utils",
        "//xla/tsl/platform:env",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:per_thread",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "cupti_pm_sampler_factory",
    srcs = ["cupti_pm_sampler_factory.cc"],
    hdrs = [
        "cupti_pm_sampler.h",
        "cupti_pm_sampler_factory.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@local_config_cuda//cuda:cupti_headers",
    ] + if_cuda_newer_than(
        "12_6",
        [":cupti_pm_sampler_impl"],
        [":cupti_pm_sampler_stub"],
    ),
)

cc_library(
    name = "cupti_pm_sampler_stub",
    srcs = ["cupti_pm_sampler_stub.cc"],
    hdrs = [
        "cupti_pm_sampler.h",
        "cupti_pm_sampler_stub.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        ":cupti_collector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "cupti_pm_sampler_impl",
    srcs = ["cupti_pm_sampler_impl.cc"],
    hdrs = [
        "cupti_pm_sampler.h",
        "cupti_pm_sampler_impl.h",
    ],
    tags = [
        "cuda-only",
        "gpu",
        "manual",  # This target requires CUDA 12.6+, therefore we only built it if it was requested via a dependency.
    ],
    deps = [
        ":cupti_collector",
        ":cupti_interface",
        ":cupti_status",
        ":cupti_utils",
        "//xla/stream_executor/cuda:cuda_status",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@local_config_cuda//cuda:cupti_headers",
        "@tsl//tsl/platform:errors",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:env",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/lib:scoped_annotation",
    ],
)

cc_library(
    name = "rocm_tracer_utils",
    srcs = ["rocm_tracer_utils.cc"],
    hdrs = ["rocm_tracer_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
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
    deps = [
        ":rocm_tracer_utils",
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/platform:status",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_rocm//rocm:rocm_headers",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocprofiler-sdk",  # buildcleaner: keep
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
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
    deps = [
        ":rocm_collector",
        ":rocm_tracer_utils",
        "//xla/stream_executor/rocm:roctracer_wrapper",
        "//xla/tsl/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@local_config_rocm//rocm:rocm_headers",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocprofiler-sdk",  # buildcleaner: keep
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:macros",
        "@tsl//tsl/platform:status",
        "@tsl//tsl/platform:types",
    ],
)

xla_cc_test(
    name = "rocm_tracer_test",
    size = "small",
    srcs = ["rocm_tracer_test.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        # Optional: only run internally if ROCm config is enabled
        "manual",
    ]),
    deps = [
        ":rocm_collector",
        ":rocm_tracer",
        ":rocm_tracer_utils",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_config_rocm//rocm:hip",  # buildcleaner: keep
        "@local_config_rocm//rocm:rocm_headers",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "rocm_collector_test",
    size = "small",
    srcs = ["rocm_collector_test.cc"],
    tags = [
        "gpu",
        "rocm-only",
    ] + if_google([
        "manual",
    ]),
    deps = [
        # ":rocm_tracer",
        ":rocm_collector",
        ":rocm_tracer_utils",
        "@com_google_googletest//:gtest_main",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "cupti_nvtx_ext_payload",
    srcs = if_cuda_newer_than(
        "13_0",
        ["cupti_nvtx_ext_payload.cc"],
        [],
    ),
    hdrs = if_cuda_newer_than(
        "13_0",
        ["cupti_nvtx_ext_payload.h"],
        [],
    ),
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/tsl/cuda:cupti",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform",
        "@tsl//tsl/platform:macros",
    ],
)

cc_library(
    name = "cupti_marker_data_parser",
    srcs = if_cuda_newer_than(
        "13_0",
        ["cupti_marker_data_parser_ext.cc"],
        ["cupti_marker_data_parser.cc"],
    ),
    hdrs = ["cupti_marker_data_parser.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/tsl/cuda:cupti",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
    ] + if_cuda_newer_than(
        "13_0",
        [":cupti_nvtx_ext_payload"],
        [],
    ),
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
        "//xla/tsl/profiler/utils:math_utils",
        "//xla/tsl/profiler/utils:parse_annotation",
        "//xla/tsl/profiler/utils:timespan",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:abi",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
        "@tsl//tsl/platform:types",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "cupti_buffer_events",
    srcs = ["cupti_buffer_events.cc"],
    hdrs = ["cupti_buffer_events.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_interface",
        ":cupti_marker_data_parser",
        ":cupti_utils",
        "//xla/tsl/cuda:cupti",
        "//xla/tsl/profiler/utils:buffer_pool",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:errors",
        "@tsl//tsl/platform:platform_port",
        "@tsl//tsl/platform:thread_annotations",
    ],
)

cc_library(
    name = "cupti_utils",
    srcs = ["cupti_utils.cc"],
    hdrs = ["cupti_utils.h"],
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings:string_view",
        "@tsl//tsl/platform:logging",
        "@tsl//tsl/platform:stringpiece",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ondevice_event_exporter",
    srcs = ["ondevice_event_exporter.cc"],
    hdrs = ["ondevice_event_exporter.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cupti_buffer_events",
        "//xla/tsl/profiler/backends/gpu:ondevice_event_collector",
        "//xla/tsl/profiler/backends/gpu:ondevice_trace_event",
        "//xla/tsl/profiler/utils:lock_free_queue",
        "//xla/tsl/profiler/utils:trace_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "//xla/tsl/profiler/utils:xplane_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

xla_cc_test(
    name = "ondevice_event_exporter_test",
    srcs = ["ondevice_event_exporter_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "no_mac",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":ondevice_event_exporter",
        "//xla/tsl/profiler/backends/gpu:ondevice_event_receiver",
        "//xla/tsl/profiler/backends/gpu:ondevice_trace_event",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "cupti_status",
    srcs = ["cupti_status.cc"],
    hdrs = ["cupti_status.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cupti_headers",
    ],
)

xla_cc_test(
    name = "cupti_buffer_events_test",
    srcs = ["cupti_buffer_events_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "no_mac",
    ],
    deps = [
        ":cupti_buffer_events",
        ":cupti_collector",  # buildcleaner: keep
        ":cupti_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "cupti_collector_test",
    size = "small",
    srcs = ["cupti_collector_test.cc"],
    tags = [
        "cuda-only",
        "gpu",
        "no_mac",
    ],
    deps = [
        ":cupti_buffer_events",
        ":cupti_collector",
        ":cupti_utils",
        "//xla/tsl/profiler/utils:xplane_builder",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cuda_library(
    name = "nvtx_with_cuda_kernels",
    testonly = 1,
    srcs = ["nvtx_with_cuda_kernels.cu.cc"],
    hdrs = ["nvtx_with_cuda_kernels.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    visibility = ["//visibility:public"],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

xla_test(
    name = "nvtx_with_cuda_kernels_test",
    size = "small",
    srcs = ["nvtx_with_cuda_kernels_test.cc"],
    backends = ["gpu"],
    tags = [
        "cuda-only",
        "no_mac",
    ],
    deps = [
        ":nvtx_with_cuda_kernels",
        "@com_google_googletest//:gtest_main",
    ],
)

cuda_library(
    name = "profile_with_cuda_kernels",
    testonly = 1,
    srcs = ["profile_with_cuda_kernels.cu.cc"],
    hdrs = ["profile_with_cuda_kernels.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = ["@local_config_cuda//cuda:cuda_headers"],
)

xla_test(
    name = "profile_with_cuda_kernels_test",
    size = "small",
    srcs = ["profile_with_cuda_kernels_test.cc"],
    args = if_google(["--heap_check="]),  # There is a memory leak in CUPTI
    backends = [
        "a100",
        "h100",
        "b200",
    ],
    tags = [
        "cuda-only",
        "no_mac",
        "nomsan",
    ],
    deps = [
        ":cupti_collector",
        ":cupti_error_manager",
        ":cupti_pm_sampler_factory",  # buildcleaner: keep
        ":cupti_tracer",
        ":cupti_utils",
        ":cupti_wrapper",
        ":profile_with_cuda_kernels",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "gpu_metadata",
    srcs = ["gpu_metadata.cc"],
    hdrs = ["gpu_metadata.h"],
    tags = [
        "cuda-only",
        "gpu",
    ],
    deps = [
        "//xla/backends/profiler/util:metadata_registry",
        "//xla/tsl/profiler/utils:xplane_schema",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
