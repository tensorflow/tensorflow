licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_gpu_library", "tf_cc_test")
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_protos_grappler",
)

filegroup(
    name = "graph_properties_testdata",
    srcs = glob([
        "graph_properties_testdata/*.pbtxt",
        "graph_properties_testdata/*.pbtxt.html",
    ]),
    visibility = ["//visibility:public"],
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_additional_all_protos",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

tf_proto_library(
    name = "op_performance_data",
    srcs = ["op_performance_data.proto"],
    cc_api_version = 2,
    default_header = True,
    protodeps = tf_additional_all_protos(),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_properties",
    srcs = ["graph_properties.cc"],
    hdrs = ["graph_properties.h"],
    visibility = ["//visibility:public"],
    copts = ["-g", "-O0"],
    deps = [
        ":utils",
        "//tensorflow/core/grappler/utils:topological_sort",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "graph_properties_test",
    srcs = ["graph_properties_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    data = [":graph_properties_testdata"],
    deps = [
        ":graph_properties",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
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

tf_gpu_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    gpu_deps = [
        "@local_config_cuda//cuda:cudnn_header",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:gpu_id",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:utils",
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
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "cost_estimator",
    hdrs = ["cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
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
        "//tensorflow/core:protos_all_cc",
    ] + tf_protos_grappler(),
)

cc_library(
    name = "virtual_scheduler",
    srcs = ["virtual_scheduler.cc"],
    hdrs = ["virtual_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
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
        "//tensorflow/core/grappler/costs:cost_estimator",
    ],
)

tf_cc_test(
    name = "virtual_scheduler_test",
    srcs = ["virtual_scheduler_test.cc"],
    deps = [
        ":virtual_placer",
        ":virtual_scheduler",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
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
)

cc_library(
    name = "op_level_cost_estimator",
    srcs = ["op_level_cost_estimator.cc"],
    hdrs = ["op_level_cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cost_estimator",
        ":op_context",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/clusters:utils",
    ] + tf_protos_grappler(),
)

tf_cc_test(
    name = "op_level_cost_estimator_test",
    srcs = ["op_level_cost_estimator_test.cc"],
    deps = [
        ":op_level_cost_estimator",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
    ] + tf_protos_grappler(),
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
