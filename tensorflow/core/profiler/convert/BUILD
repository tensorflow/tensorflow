load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow/core/profiler:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "xplane_to_op_metrics_db",
    srcs = ["xplane_to_op_metrics_db.cc"],
    hdrs = ["xplane_to_op_metrics_db.h"],
    deps = [
        ":op_metrics_db_combiner",
        ":op_stack",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:cost_utils",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
        "//tensorflow/core/profiler/utils:op_utils",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:timespan",
        "//tensorflow/core/profiler/utils:trace_utils",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_to_op_metrics_db_test",
    size = "small",
    srcs = ["xplane_to_op_metrics_db_test.cc"],
    deps = [
        ":xplane_to_op_metrics_db",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "op_metrics_db_combiner",
    srcs = ["op_metrics_db_combiner.cc"],
    hdrs = ["op_metrics_db_combiner.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
    ],
)

cc_library(
    name = "op_metrics_to_record",
    srcs = ["op_metrics_to_record.cc"],
    hdrs = ["op_metrics_to_record.h"],
    deps = [
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/utils:math_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "op_stack",
    hdrs = ["op_stack.h"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "op_stats_to_overview_page",
    srcs = ["op_stats_to_overview_page.cc"],
    hdrs = ["op_stats_to_overview_page.h"],
    deps = [
        ":op_metrics_to_record",
        ":op_stats_to_input_pipeline_analysis",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "//tensorflow/core/profiler/protobuf:input_pipeline_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:overview_page_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_function_proto_cc",
        "//tensorflow/core/profiler/utils:errors",
        "//tensorflow/core/profiler/utils:html_utils",
        "//tensorflow/core/profiler/utils:math_utils",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "op_stats_to_input_pipeline_analysis",
    srcs = ["op_stats_to_input_pipeline_analysis.cc"],
    hdrs = ["op_stats_to_input_pipeline_analysis.h"],
    deps = [
        ":op_metrics_to_record",
        ":step_events_to_steps_db",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "//tensorflow/core/profiler/protobuf:input_pipeline_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "//tensorflow/core/profiler/utils:errors",
        "//tensorflow/core/profiler/utils:event_span",
        "//tensorflow/core/profiler/utils:html_utils",
        "//tensorflow/core/profiler/utils:math_utils",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/core/util:stats_calculator_portable",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "op_stats_to_tf_stats",
    srcs = ["op_stats_to_tf_stats.cc"],
    hdrs = ["op_stats_to_tf_stats.h"],
    deps = [
        ":op_metrics_to_record",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_stats_proto_cc",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
        "//tensorflow/core/profiler/utils:time_utils",
    ],
)

tf_cc_test(
    name = "op_stats_to_tf_stats_test",
    size = "small",
    srcs = ["op_stats_to_tf_stats_test.cc"],
    deps = [
        ":op_stats_to_tf_stats",
        ":xplane_to_op_stats",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:op_metrics_db_utils",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "step_events_to_steps_db",
    srcs = ["step_events_to_steps_db.cc"],
    hdrs = ["step_events_to_steps_db.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "//tensorflow/core/profiler/utils:event_span",
        "//tensorflow/core/profiler/utils:timespan",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "trace_events_to_json",
    srcs = ["trace_events_to_json.cc"],
    hdrs = ["trace_events_to_json.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@jsoncpp_git//:jsoncpp",
    ],
)

tf_cc_test(
    name = "trace_events_to_json_test",
    srcs = ["trace_events_to_json_test.cc"],
    deps = [
        ":trace_events_to_json",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "xplane_to_op_stats",
    srcs = ["xplane_to_op_stats.cc"],
    hdrs = ["xplane_to_op_stats.h"],
    deps = [
        ":op_metrics_db_combiner",
        ":step_events_to_steps_db",
        ":xplane_to_kernel_stats_db",
        ":xplane_to_op_metrics_db",
        ":xplane_to_step_events",
        ":xplane_to_tf_functions",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_function_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:event_span",
        "//tensorflow/core/profiler/utils:hardware_type_utils",
        "//tensorflow/core/profiler/utils:kernel_stats_utils",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "xplane_to_op_stats_test",
    size = "small",
    srcs = ["xplane_to_op_stats_test.cc"],
    deps = [
        ":xplane_to_op_stats",
        ":xplane_to_tf_functions",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/profiler/protobuf:op_metrics_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:steps_db_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_function_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:group_events",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_to_profile_response",
    srcs = ["xplane_to_profile_response.cc"],
    hdrs = ["xplane_to_profile_response.h"],
    deps = [
        ":op_stats_to_input_pipeline_analysis",
        ":op_stats_to_overview_page",
        ":op_stats_to_tf_stats",
        ":trace_events_to_json",
        ":xplane_to_memory_profile",
        ":xplane_to_op_stats",
        ":xplane_to_trace_events",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/protobuf:hardware_types_proto_cc",
        "//tensorflow/core/profiler/protobuf:input_pipeline_proto_cc",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:memory_profile_proto_cc",
        "//tensorflow/core/profiler/protobuf:op_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:overview_page_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/rpc/client:save_profile",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "xplane_to_profile_response_test",
    size = "small",
    srcs = ["xplane_to_profile_response_test.cc"],
    deps = [
        ":xplane_to_profile_response",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler:profiler_service_proto_cc",
        "//tensorflow/core/profiler/protobuf:input_pipeline_proto_cc",
        "//tensorflow/core/profiler/protobuf:overview_page_proto_cc",
        "//tensorflow/core/profiler/protobuf:tf_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:group_events",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_to_step_events",
    srcs = ["xplane_to_step_events.cc"],
    hdrs = ["xplane_to_step_events.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:event_span",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:timespan",
        "//tensorflow/core/profiler/utils:trace_utils",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_to_step_events_test",
    size = "small",
    srcs = ["xplane_to_step_events_test.cc"],
    deps = [
        ":xplane_to_step_events",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:event_span",
        "//tensorflow/core/profiler/utils:group_events",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "xplane_to_trace_events",
    srcs = ["xplane_to_trace_events.cc"],
    hdrs = ["xplane_to_trace_events.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_to_trace_events_test",
    size = "small",
    srcs = ["xplane_to_trace_events_test.cc"],
    deps = [
        ":xplane_to_trace_events",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/profiler/protobuf:trace_events_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ],
)

cc_library(
    name = "xplane_to_kernel_stats_db",
    srcs = ["xplane_to_kernel_stats_db.cc"],
    hdrs = ["xplane_to_kernel_stats_db.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:kernel_stats_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:kernel_stats_utils",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:trace_utils",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "xplane_to_tf_functions",
    srcs = ["xplane_to_tf_functions.cc"],
    hdrs = ["xplane_to_tf_functions.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/profiler/protobuf:tf_function_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:math_utils",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:timespan",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_to_tf_functions_test",
    size = "small",
    srcs = ["xplane_to_tf_functions_test.cc"],
    deps = [
        ":xplane_to_tf_functions",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/profiler/protobuf:tf_function_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "xplane_to_memory_profile",
    srcs = ["xplane_to_memory_profile.cc"],
    hdrs = ["xplane_to_memory_profile.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/framework:protos_all_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/core/profiler/protobuf:memory_profile_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:tf_xplane_visitor",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "xplane_to_memory_profile_test",
    size = "small",
    srcs = ["xplane_to_memory_profile_test.cc"],
    deps = [
        ":xplane_to_memory_profile",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:memory_profile_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/strings",
    ],
)
