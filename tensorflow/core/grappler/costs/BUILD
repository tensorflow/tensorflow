load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
    "tf_cuda_library",
)
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_proto_library",
    "tf_protos_grappler",
    "tf_pyclif_proto_library",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

alias(
    name = "graph_properties_testdata",
    actual = "//tensorflow/core/grappler/costs/graph_properties_testdata:graph_properties_testdata",
    visibility = ["//visibility:public"],
)

tf_proto_library(
    name = "op_performance_data",
    srcs = ["op_performance_data.proto"],
    cc_api_version = 2,
    make_default_target_header_only = True,
    protodeps = [
        "//tensorflow/core/framework:attr_value_proto",
        "//tensorflow/core/framework:resource_handle_proto",
        "//tensorflow/core/framework:tensor_proto",
        "//tensorflow/core/framework:tensor_shape_proto",
        "//tensorflow/core/framework:types_proto",
        "//tensorflow/core/protobuf:for_core_protos",
    ],
    visibility = ["//visibility:public"],
)

tf_pyclif_proto_library(
    name = "op_performance_data_pyclif",
    proto_lib = ":op_performance_data",
    proto_srcfile = "op_performance_data.proto",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "analytical_cost_estimator.h",
        "cost_estimator.h",
        "graph_memory.h",
        "graph_properties.h",
        "measuring_cost_estimator.h",
        "op_context.h",
        "op_level_cost_estimator.h",
        "utils.h",
        "virtual_placer.h",
        "virtual_scheduler.h",
    ],
    visibility = [
        "//tensorflow/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "graph_properties",
    srcs = ["graph_properties.cc"],
    hdrs = ["graph_properties.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/types:optional",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:mutable_graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:evaluation_utils",
        "//tensorflow/core/grappler/utils:functions",
        "//tensorflow/core/grappler/utils:topological_sort",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "graph_properties_test",
    srcs = ["graph_properties_test.cc"],
    args = ["--heap_check="],  # The GPU tracer leaks memory. TODO(b/185483595): use a dependency instead of a flag
    data = [":graph_properties_testdata"],
    tags = [
        "nomsan",  # TODO(b/160921160): broken by NOAUTOROLLBACK CL
    ],
    deps = [
        ":graph_properties",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/graph:mkl_graph_util",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "//tensorflow/core/grappler/inputs:utils",
    ],
)

cc_library(
    name = "graph_memory",
    srcs = ["graph_memory.cc"],
    hdrs = ["graph_memory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":graph_properties",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
    ],
)

tf_cc_test(
    name = "graph_memory_test",
    srcs = ["graph_memory_test.cc"],
    args = ["--heap_check="],  # The GPU tracer leaks memory. TODO(b/185483595): use a dependency instead of a flag
    deps = [
        ":graph_memory",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "robust_stats",
    srcs = ["robust_stats.cc"],
    hdrs = ["robust_stats.h"],
    visibility = ["//visibility:public"],
)

tf_cc_test(
    name = "robust_stats_test",
    srcs = ["robust_stats_test.cc"],
    deps = [
        ":robust_stats",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cuda_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:utils",
        "//tensorflow/core/util:overflow",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/framework:tensor_testutil",
    ],
)

cc_library(
    name = "cost_estimator",
    srcs = ["cost_estimator.cc"],
    hdrs = ["cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

tf_cc_test(
    name = "cost_estimator_test",
    srcs = ["cost_estimator_test.cc"],
    deps = [
        ":cost_estimator",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "virtual_placer",
    srcs = ["virtual_placer.cc"],
    hdrs = ["virtual_placer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler/clusters:cluster",
    ],
)

tf_cc_test(
    name = "virtual_placer_test",
    srcs = ["virtual_placer_test.cc"],
    deps = [
        ":virtual_placer",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
    ],
)

cc_library(
    name = "op_context",
    hdrs = ["op_context.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core:protos_all_cc",
    ] + tf_protos_grappler(),
)

cc_library(
    name = "virtual_scheduler",
    srcs = ["virtual_scheduler.cc"],
    hdrs = ["virtual_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":graph_properties",
        ":op_context",
        ":utils",
        ":virtual_placer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:utils",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_cc_test(
    name = "virtual_scheduler_test",
    srcs = ["virtual_scheduler_test.cc"],
    deps = [
        ":graph_properties",
        ":utils",
        ":virtual_placer",
        ":virtual_scheduler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "measuring_cost_estimator",
    srcs = ["measuring_cost_estimator.cc"],
    hdrs = ["measuring_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":robust_stats",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/costs:cost_estimator",
        "//tensorflow/core/kernels:ops_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "op_level_cost_estimator",
    srcs = ["op_level_cost_estimator.cc"],
    hdrs = ["op_level_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":op_context",
        ":utils",
        "@com_google_absl//absl/strings",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/clusters:utils",
        "//tensorflow/core/util:overflow",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "op_level_cost_estimator_test",
    srcs = ["op_level_cost_estimator_test.cc"],
    tags = ["no_oss"],  # b/163222310
    deps = [
        ":op_level_cost_estimator",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
    ],
)

cc_library(
    name = "analytical_cost_estimator",
    srcs = ["analytical_cost_estimator.cc"],
    hdrs = ["analytical_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":graph_properties",
        ":op_level_cost_estimator",
        ":utils",
        ":virtual_placer",
        ":virtual_scheduler",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/util:overflow",
    ] + tf_protos_grappler(),
    alwayslink = 1,
)

tf_cc_test(
    name = "analytical_cost_estimator_test",
    srcs = ["analytical_cost_estimator_test.cc"],
    deps = [
        ":analytical_cost_estimator",
        ":virtual_scheduler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
    ],
)

# copybara:uncomment_begin(google-only)
# py_proto_library(
#     name = "op_performance_data_py_pb2",
#     has_services = 0,
#     api_version = 2,
#     visibility = ["//visibility:public"],
#     deps = [":op_performance_data"],
# )
# copybara:uncomment_end
