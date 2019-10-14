load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_library")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "op_types",
    srcs = ["op_types.cc"],
    hdrs = ["op_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    deps = [
        ":grappler_item",
        ":utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

tf_cuda_library(
    name = "devices",
    srcs = ["devices.cc"],
    hdrs = ["devices.h"],
    cuda_deps = [
        "//tensorflow/core:gpu_init",
        "//tensorflow/core:stream_executor",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "graph_topology_view",
    srcs = ["graph_topology_view.cc"],
    hdrs = ["graph_topology_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "graph_topology_view_test",
    srcs = ["graph_topology_view_test.cc"],
    deps = [
        ":graph_topology_view",
        ":graph_view",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "graph_view",
    srcs = ["graph_view.cc"],
    hdrs = ["graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "graph_view_test",
    srcs = ["graph_view_test.cc"],
    deps = [
        ":graph_view",
        ":grappler_item",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grappler_item",
    srcs = [
        "grappler_item.cc",
    ],
    hdrs = ["grappler_item.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":op_types",
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "grappler_item_builder",
    srcs = [
        "grappler_item_builder.cc",
    ],
    hdrs = ["grappler_item_builder.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":grappler_item",
        ":op_types",
        ":utils",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/costs:graph_properties",
        "//tensorflow/core/grappler/inputs:utils",
        "//tensorflow/core/grappler/optimizers:model_pruner",
    ],
)

tf_cc_test(
    name = "grappler_item_test",
    srcs = ["grappler_item_test.cc"],
    deps = [
        ":grappler_item",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

tf_cc_test(
    name = "grappler_item_builder_test",
    srcs = ["grappler_item_builder_test.cc"],
    deps = [
        ":grappler_item_builder",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:grad_testutil",
        "//tensorflow/cc:gradients",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "mutable_graph_view",
    srcs = [
        "mutable_graph_view.cc",
    ],
    hdrs = ["mutable_graph_view.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_view",
        ":op_types",
        ":utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_cc_test(
    name = "mutable_graph_view_test",
    srcs = ["mutable_graph_view_test.cc"],
    deps = [
        ":grappler_item",
        ":mutable_graph_view",
        ":utils",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
