licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_protos_all")

cc_library(
    name = "function_rename",
    srcs = ["function_rename.cc"],
    hdrs = [
        "function_rename.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "function_rename_test",
    srcs = ["function_rename_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":function_rename",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ] + tf_protos_all(),
)

cc_library(
    name = "graph_utils",
    srcs = ["graph_utils.cc"],
    hdrs = [
        "graph_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "graph_utils_test",
    srcs = ["graph_utils_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "map_and_batch_fusion",
    srcs = ["map_and_batch_fusion.cc"],
    hdrs = [
        "map_and_batch_fusion.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "map_and_batch_fusion_test",
    srcs = ["map_and_batch_fusion_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        ":map_and_batch_fusion",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "noop_elimination",
    srcs = ["noop_elimination.cc"],
    hdrs = [
        "noop_elimination.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "noop_elimination_test",
    srcs = ["noop_elimination_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        ":noop_elimination",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "shuffle_and_repeat_fusion",
    srcs = ["shuffle_and_repeat_fusion.cc"],
    hdrs = [
        "shuffle_and_repeat_fusion.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core/grappler:graph_view",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:op_types",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer",
        "//tensorflow/core/grappler/optimizers:custom_graph_optimizer_registry",
    ] + tf_protos_all(),
)

tf_cc_test(
    name = "shuffle_and_repeat_fusion_test",
    srcs = ["shuffle_and_repeat_fusion_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_utils",
        ":shuffle_and_repeat_fusion",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
    ],
)

cc_library(
    name = "data",
    visibility = ["//visibility:public"],
    deps = [
        ":function_rename",
        ":map_and_batch_fusion",
        ":noop_elimination",
        ":shuffle_and_repeat_fusion",
    ],
    alwayslink = 1,
)
