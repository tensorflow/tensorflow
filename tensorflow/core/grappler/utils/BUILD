licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "scc",
    srcs = ["scc.cc"],
    hdrs = ["scc.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:utils",
    ],
)

tf_cc_test(
    name = "scc_test",
    srcs = ["scc_test.cc"],
    data = [
        "//tensorflow/core/grappler/costs:graph_properties_testdata",
    ],
    deps = [
        ":scc",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "topological_sort",
    srcs = ["topological_sort.cc"],
    hdrs = ["topological_sort.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "topological_sort_test",
    srcs = ["topological_sort_test.cc"],
    deps = [
        ":topological_sort",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "frame",
    srcs = ["frame.cc"],
    hdrs = ["frame.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "frame_test",
    size = "small",
    srcs = ["frame_test.cc"],
    deps = [
        ":frame",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "traversal",
    srcs = ["traversal.cc"],
    hdrs = ["traversal.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "traversal_test",
    srcs = ["traversal_test.cc"],
    deps = [
        ":traversal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "grappler_test",
    testonly = 1,
    srcs = [
        "grappler_test.cc",
    ],
    hdrs = ["grappler_test.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "grappler_test_test",
    size = "small",
    srcs = ["grappler_test_test.cc"],
    deps = [
        ":grappler_test",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:direct_session",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:utils",
    ],
)

cc_library(
    name = "functions",
    srcs = [
        "functions.cc",
    ],
    hdrs = ["functions.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "functions_test",
    srcs = ["functions_test.cc"],
    deps = [
        ":functions",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "colocation",
    srcs = ["colocation.cc"],
    hdrs = ["colocation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:utils",
    ],
)

tf_cc_test(
    name = "colocation_test",
    size = "small",
    srcs = ["colocation_test.cc"],
    deps = [
        ":colocation",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
