# Libraries for performance modeling of HLO.
load("//xla/tests:build_defs.bzl", "xla_test")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//xla:xla.bzl", "xla_cc_test", "xla_nvml_deps")
load("@local_tsl//tsl:tsl.default.bzl", "get_compatible_with_portable")
load("@local_tsl//tsl/platform:build_config.bzl", "tf_proto_library")
load("@local_tsl//tsl/platform:build_config_root.bzl", "tf_cuda_tests_tags")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "analytical_latency_estimator",
    srcs = ["analytical_latency_estimator.cc"],
    hdrs = ["analytical_latency_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:latency_hiding_scheduler",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/time",
    ],
)

xla_test(
    name = "analytical_latency_estimator_test",
    srcs = ["analytical_latency_estimator_test.cc"],
    backend_tags = {"gpu": [
        "requires-gpu-sm70",
    ]},
    backends = [
        "gpu",
    ],
    deps = [
        ":analytical_latency_estimator",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:latency_hiding_scheduler",
        "//xla/service/gpu/tests:gpu_codegen_test",
        "//xla/stream_executor:device_description",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/platform:statusor",
        "@local_tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "gpu_cost_model_stats_collection",
    srcs = ["gpu_cost_model_stats_collection.cc"],
    hdrs = ["gpu_cost_model_stats_collection.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//xla:statusor",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_graph_dumper",
        "//xla/service:hlo_pass",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:status",
    ],
)

xla_cc_test(
    name = "gpu_cost_model_stats_collection_test",
    srcs = ["gpu_cost_model_stats_collection_test.cc"],
    deps = [
        ":gpu_cost_model_stats_collection",
        ":gpu_hlo_cost_analysis",
        "//xla:shape_util",
        "//xla/service:hlo_cost_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/tests:hlo_test_base",
        "//xla/tests:verified_hlo_module",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/lib/core:status_test_util",
    ],
)

cc_library(
    name = "gpu_hlo_cost_analysis",
    srcs = ["gpu_hlo_cost_analysis.cc"],
    hdrs = [
        "gpu_hlo_cost_analysis.h",
        "hlo_op_profiles.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = ["//visibility:public"],
    deps = [
        ":hlo_op_profile_proto_cc",
        "//xla:shape_util",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/service:collective_ops_utils",
        "//xla/service:elemental_ir_emitter",
        "//xla/service:hlo_cost_analysis",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:cublas_cudnn",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

xla_cc_test(
    name = "gpu_hlo_cost_analysis_test",
    srcs = ["gpu_hlo_cost_analysis_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "gpu_performance_model",
    srcs = ["gpu_performance_model.cc"],
    hdrs = ["gpu_performance_model.h"],
    local_defines = if_cuda_is_configured(["GOOGLE_CUDA=1"]),
    visibility = ["//visibility:public"],
    deps = [
        ":gpu_hlo_cost_analysis",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_dataflow_analysis",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_fusible",
        "//xla/service/gpu:hlo_fusion_analysis",
        "//xla/service/gpu:hlo_traversal",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/time",
    ] + if_cuda_is_configured(xla_nvml_deps()),
)

xla_cc_test(
    name = "gpu_performance_model_test",
    srcs = ["gpu_performance_model_test.cc"],
    deps = [
        ":gpu_hlo_cost_analysis",
        ":gpu_performance_model",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/gpu:gpu_device_info_for_tests",
        "//xla/stream_executor:device_description",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tf_proto_library(
    name = "hlo_op_profile_proto",
    srcs = ["hlo_op_profile.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//xla/service:hlo_proto",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "hlo_op_profiler_lib",
    testonly = True,
    srcs = ["hlo_op_profiler.cc"],
    hdrs = ["hlo_op_profiler.h"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    visibility = ["//visibility:public"],
    deps = [
        ":hlo_op_profile_proto_cc",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/backends/profiler/gpu:cupti_collector",
        "//xla/backends/profiler/gpu:cupti_tracer",
        "//xla/hlo/ir:hlo",
        "//xla/service:executable",
        "//xla/service:gpu_plugin",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_runner",
        "//xla/service:interpreter_plugin",
        "//xla/stream_executor:device_description",
        "//xla/tests:test_utils",
        "@com_google_absl//absl/time",
        "@local_tsl//tsl/platform:errors",
    ],
)

xla_cc_test(
    name = "hlo_op_profiler_run",
    timeout = "eternal",
    srcs = ["hlo_op_profiler_run.cc"],
    # Disable backend optimizations (in particular reassociate and instcombine) which would optimize
    # expressions like integer add and multiply.
    args = ["--xla_backend_optimization_level=0"],
    # This is a development tool, not a normal test, and thus should only be run
    # manually with --config=cuda.
    tags = [
        "gpu",
        "manual",
        "notap",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":hlo_op_profile_proto_cc",
        ":hlo_op_profiler_lib",
        "//xla:debug_options_flags",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_runner",
        "//xla/service:platform_util",
        "//xla/stream_executor:device_description",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_tsl//tsl/platform:env",
        "@local_tsl//tsl/platform:path",
        "@local_tsl//tsl/platform:platform_port",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/util:command_line_flags",
    ],
)

xla_cc_test(
    name = "hlo_op_profiler_test",
    srcs = ["hlo_op_profiler_test.cc"],
    local_defines = if_cuda(["GOOGLE_CUDA"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":hlo_op_profiler_lib",
        "//xla/hlo/ir:hlo",
        "//xla/service:gpu_plugin",
        "//xla/tests:hlo_test_base",
        "@local_tsl//tsl/platform:test_main",
    ],
)
