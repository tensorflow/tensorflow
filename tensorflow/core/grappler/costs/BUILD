licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

tf_proto_library(
    name = "op_performance_data",
    srcs = ["op_performance_data.proto"],
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "graph_properties",
    srcs = ["graph_properties.cc"],
    hdrs = ["graph_properties.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_performance_data_cc",
        ":utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
    ],
)

cc_test(
    name = "graph_properties_test",
    srcs = ["graph_properties_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    deps = [
        ":graph_properties",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:single_machine",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "graph_memory",
    srcs = ["graph_memory.cc"],
    hdrs = ["graph_memory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_properties",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:cluster",
    ],
)

cc_test(
    name = "graph_memory_test",
    srcs = ["graph_memory_test.cc"],
    args = ["--heap_check=local"],  # The GPU tracer leaks memory
    deps = [
        ":graph_memory",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_performance_data_cc",
        "//third_party/eigen3",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + if_cuda([
        "//tensorflow/core:cuda",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
)

cc_library(
    name = "cost_estimator",
    hdrs = ["cost_estimator.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
    ],
)
