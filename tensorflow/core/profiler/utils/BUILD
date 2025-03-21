load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_library")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/core/profiler:friends",
    ],
)

cc_library(
    name = "diagnostics",
    srcs = ["diagnostics.cc"],
    hdrs = ["diagnostics.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:diagnostics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "event_span",
    srcs = ["event_span.cc"],
    hdrs = ["event_span.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "hardware_type_utils",
    srcs = ["hardware_type_utils.cc"],
    hdrs = ["hardware_type_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":xplane_schema",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
    ],
)

tf_cc_test(
    name = "hardware_type_utils_test",
    srcs = ["hardware_type_utils_test.cc"],
    deps = [
        ":hardware_type_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "math_utils",
    hdrs = ["math_utils.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
    ],
)

cc_library(
    name = "html_utils",
    hdrs = ["html_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_metrics_db_utils",
    srcs = ["op_metrics_db_utils.cc"],
    hdrs = ["op_metrics_db_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":xplane_visitor",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

tf_cc_test(
    name = "op_metrics_db_utils_test",
    srcs = ["op_metrics_db_utils_test.cc"],
    deps = [
        ":op_metrics_db_utils",
        ":xplane_builder",
        ":xplane_schema",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "op_utils",
    srcs = ["op_utils.cc"],
    hdrs = ["op_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":hlo_module_map",
        ":op_metrics_db_utils",
        "//tensorflow/core/profiler/convert:op_metrics_db_combiner",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tsl/platform:types",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "trace_utils",
    hdrs = ["trace_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "@local_xla//xla/tsl/profiler/utils:trace_utils",
    ],
)

cc_library(
    name = "xplane_builder",
    hdrs = ["xplane_builder.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@local_xla//xla/tsl/profiler/utils:xplane_builder",
    ],
)

cc_library(
    name = "xplane_schema",
    hdrs = ["xplane_schema.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = ["@local_xla//xla/tsl/profiler/utils:xplane_schema"],
)

cc_library(
    name = "xplane_utils",
    hdrs = ["xplane_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@local_xla//xla/tsl/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_test_utils",
    testonly = True,
    hdrs = ["xplane_test_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = ["@local_xla//xla/tsl/profiler/utils:xplane_test_utils"],
)

cc_library(
    name = "xplane_visitor",
    hdrs = ["xplane_visitor.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
    ],
)

cc_library(
    name = "cost_utils",
    srcs = ["cost_utils.cc"],
    hdrs = ["cost_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/grappler/costs:op_context",
        "//tensorflow/core/grappler/costs:op_level_cost_estimator",
        "//tensorflow/core/grappler/costs:op_performance_data_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
    ],
)

cc_library(
    name = "host_offload_utils",
    srcs = ["host_offload_utils.cc"],
    hdrs = ["host_offload_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/tsl/profiler/utils:timespan",
    ],
)

cc_library(
    name = "derived_timeline",
    srcs = ["derived_timeline.cc"],
    hdrs = ["derived_timeline.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":gpu_event_stats",
        ":hlo_module_map",
        ":hlo_proto_map",
        ":host_offload_utils",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
        "@local_xla//xla/tsl/profiler/utils:device_utils",
        "@local_xla//xla/tsl/profiler/utils:group_events",
        "@local_xla//xla/tsl/profiler/utils:math_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_op_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:timespan",
        "@local_xla//xla/tsl/profiler/utils:tpu_xplane_utils",
        "@local_xla//xla/tsl/profiler/utils:trace_utils",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
        "@local_xla//xla/tsl/profiler/utils:xplane_visitor",
        "@local_xla//xla/tsl/util:stats_calculator_portable",
    ],
)

tf_cc_test(
    name = "derived_timeline_test",
    srcs = ["derived_timeline_test.cc"],
    deps = [
        ":derived_timeline",
        ":trace_utils",
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_test_utils",
        ":xplane_visitor",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/utils:group_events",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
        "@local_xla//xla/tsl/profiler/utils:xplane_schema",
    ],
)

cc_library(
    name = "kernel_stats_utils",
    srcs = ["kernel_stats_utils.cc"],
    hdrs = ["kernel_stats_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "kernel_stats_utils_test",
    srcs = ["kernel_stats_utils_test.cc"],
    deps = [
        ":kernel_stats_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/backends/profiler/gpu:cupti_buffer_events",
    ],
)

cc_library(
    name = "tfstreamz_utils",
    srcs = ["tfstreamz_utils.cc"],
    hdrs = ["tfstreamz_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":xplane_builder",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/profiler/protobuf:tfstreamz_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "step_intersection",
    srcs = ["step_intersection.cc"],
    hdrs = ["step_intersection.h"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:types",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/profiler/utils:timespan",
    ],
)

tf_cc_test(
    name = "step_intersection_test",
    srcs = ["step_intersection_test.cc"],
    deps = [
        ":step_intersection",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "device_caps_utils",
    srcs = ["device_caps_utils.cc"],
    hdrs = ["device_caps_utils.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":xplane_builder",
        ":xplane_schema",
        ":xplane_visitor",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
    ],
)

cc_library(
    name = "gpu_event_stats",
    srcs = ["gpu_event_stats.cc"],
    hdrs = ["gpu_event_stats.h"],
    copts = tf_profiler_copts(),
    visibility = [":friends"],
    deps = [
        ":xplane_schema",
        ":xplane_visitor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_proto_map",
    srcs = ["hlo_proto_map.cc"],
    hdrs = ["hlo_proto_map.h"],
    visibility = [":friends"],
    deps = [
        ":xplane_schema",
        ":xplane_utils",
        ":xplane_visitor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
        "@local_xla//xla/tsl/profiler/utils:tf_xplane_visitor",
    ],
)

cc_library(
    name = "hlo_proto_to_module",
    srcs = ["hlo_proto_to_module.cc"],
    hdrs = ["hlo_proto_to_module.h"],
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cuda_library(
    name = "hlo_module_map",
    srcs = ["hlo_module_map.cc"],
    hdrs = ["hlo_module_map.h"],
    cuda_deps = [
        "@local_xla//xla/service/gpu/model:gpu_hlo_cost_analysis",
    ],
    visibility = [":friends"],
    deps = [
        ":hlo_module_utils",
        ":hlo_proto_map",
        ":hlo_proto_to_module",
        "//tensorflow/core/platform:path",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/service:hlo_proto_cc",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
    ],
)

cc_library(
    name = "hlo_module_utils",
    hdrs = ["hlo_module_utils.h"],
    visibility = [
        ":friends",
        # copybara:uncomment "//tensorflow/compiler/mlir/lite/experimental/google/tooling/google:__subpackages__",
    ],
    deps = [
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tsl/profiler/convert:xla_op_utils",
    ],
)

tf_cc_test(
    name = "hlo_module_utils_test",
    srcs = ["hlo_module_utils_test.cc"],
    deps = [
        ":hlo_module_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/tests:hlo_test_base",
    ],
)

cc_library(
    name = "xprof_gpu_cost_analysis",
    srcs = ["xprof_gpu_cost_analysis.cc"],
    hdrs = ["xprof_gpu_cost_analysis.h"],
    visibility = [":friends"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/service/gpu:cublas_cudnn",
        "@local_xla//xla/service/gpu/model:gpu_hlo_cost_analysis",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tpu_step_breakdown_utils",
    hdrs = ["tpu_step_breakdown_utils.h"],
    visibility = [":friends"],
    deps = ["//tensorflow/core/profiler/protobuf:steps_db_proto_cc"],
)

cc_library(
    name = "tpu_step_details_utils",
    hdrs = ["tpu_step_details_utils.h"],
    visibility = [":friends"],
    deps = ["//tensorflow/core/profiler/protobuf:tpu_input_pipeline_proto_cc"],
)

tf_cc_test(
    name = "xprof_gpu_cost_analysis_test",
    srcs = ["xprof_gpu_cost_analysis_test.cc"],
    deps = [
        ":xprof_gpu_cost_analysis",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/testlib:test_helpers",
        "@local_xla//xla/service:hlo_cost_analysis",
        "@local_xla//xla/tests:hlo_test_base",
        "@local_xla//xla/tests:xla_internal_test_main",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)
