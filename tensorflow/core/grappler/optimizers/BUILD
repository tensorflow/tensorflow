licenses(["notice"])  # Apache 2.0

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

filegroup(
    name = "android_srcs",
    srcs = glob(
        [
            "*_optimizer.*",
            "constant_folding.*",
            "model_pruner.*",
            "graph_rewriter.*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "constant_folding",
    srcs = ["constant_folding.cc"],
    hdrs = [
        "constant_folding.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
    ],
)

cc_test(
    name = "constant_folding_test",
    srcs = ["constant_folding_test.cc"],
    deps = [
        ":constant_folding",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "graph_rewriter",
    srcs = ["graph_rewriter.cc"],
    hdrs = [
        "graph_rewriter.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
    ],
)

cc_library(
    name = "graph_optimizer",
    hdrs = [
        "graph_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "model_pruner",
    srcs = ["model_pruner.cc"],
    hdrs = [
        "model_pruner.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        ":graph_rewriter",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
    ],
)

cc_test(
    name = "model_pruner_test",
    srcs = ["model_pruner_test.cc"],
    deps = [
        ":model_pruner",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "memory_optimizer",
    srcs = ["memory_optimizer.cc"],
    hdrs = [
        "memory_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        ":graph_rewriter",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
    ],
)

cc_test(
    name = "memory_optimizer_test",
    srcs = ["memory_optimizer_test.cc"],
    deps = [
        ":memory_optimizer",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)

cc_library(
    name = "layout_optimizer",
    srcs = ["layout_optimizer.cc"],
    hdrs = [
        "layout_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":graph_optimizer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:devices",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/clusters:cluster",
    ],
)

cc_library(
    name = "meta_optimizer",
    srcs = ["meta_optimizer.cc"],
    hdrs = [
        "meta_optimizer.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_folding",
        ":graph_optimizer",
        ":layout_optimizer",
        ":model_pruner",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler:grappler_item",
    ],
)
