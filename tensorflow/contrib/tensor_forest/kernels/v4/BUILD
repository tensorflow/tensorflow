# TensorFlow code for training random forests.

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core:platform/default/build_config_root.bzl", "if_static")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

DECISION_TREE_RESOURCE_DEPS = [
    ":decision_node_evaluator",
    ":input_data",
    ":leaf_model_operators",
    "//tensorflow/core:framework_headers_lib",
] + if_static(
    [
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
        "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
    ],
    [
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
        "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
    ],
)

cc_library(
    name = "decision-tree-resource",
    hdrs = ["decision-tree-resource.h"],
    deps = DECISION_TREE_RESOURCE_DEPS + if_static([":decision-tree-resource_impl"]),
)

cc_library(
    name = "decision-tree-resource_impl",
    srcs = ["decision-tree-resource.cc"],
    hdrs = ["decision-tree-resource.h"],
    deps = DECISION_TREE_RESOURCE_DEPS,
)

cc_library(
    name = "fertile-stats-resource",
    srcs = ["fertile-stats-resource.cc"],
    hdrs = ["fertile-stats-resource.h"],
    deps = [
        ":decision_node_evaluator",
        ":input_data",
        ":input_target",
        ":leaf_model_operators",
        ":split_collection_operators",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
)

cc_library(
    name = "input_data",
    srcs = ["input_data.cc"],
    hdrs = ["input_data.h"],
    deps = [
        "//tensorflow/contrib/tensor_forest:tree_utils",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc_headers_only",
        ],
    ),
)

cc_library(
    name = "input_target",
    hdrs = ["input_target.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "leaf_model_operators",
    srcs = ["leaf_model_operators.cc"],
    hdrs = ["leaf_model_operators.h"],
    deps = [
        ":input_target",
        ":params",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
)

tf_cc_test(
    name = "leaf_model_operators_test",
    srcs = ["leaf_model_operators_test.cc"],
    deps = [
        ":leaf_model_operators",
        ":test_utils",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
        "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
        "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        "//tensorflow/core",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "grow_stats",
    srcs = ["grow_stats.cc"],
    hdrs = ["grow_stats.h"],
    deps = [
        ":decision_node_evaluator",
        ":input_data",
        ":input_target",
        ":params",
        ":stat_utils",
        "//tensorflow/contrib/tensor_forest:tree_utils",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
)

tf_cc_test(
    name = "grow_stats_test",
    srcs = ["grow_stats_test.cc"],
    deps = [
        ":grow_stats",
        ":test_utils",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
        "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
        "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        "//tensorflow/core",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "candidate_graph_runner",
    srcs = ["candidate_graph_runner.cc"],
    hdrs = ["candidate_graph_runner.h"],
    deps = [
        ":input_data",
        ":input_target",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
        ],
    ),
)

cc_library(
    name = "decision_node_evaluator",
    srcs = ["decision_node_evaluator.cc"],
    hdrs = ["decision_node_evaluator.h"],
    deps = [
        ":input_data",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc_headers_only",
        ],
    ),
)

tf_cc_test(
    name = "decision_node_evaluator_test",
    srcs = ["decision_node_evaluator_test.cc"],
    deps = [
        ":decision_node_evaluator",
        ":test_utils",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
        "//tensorflow/core",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "split_collection_operators",
    srcs = ["split_collection_operators.cc"],
    hdrs = ["split_collection_operators.h"],
    deps = [
        ":grow_stats",
        ":input_data",
        ":input_target",
        ":leaf_model_operators",
        ":params",
        ":stat_utils",
        "//tensorflow/contrib/tensor_forest:tree_utils",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
)

cc_library(
    name = "graph_collection_operator",
    srcs = ["graph_collection_operator.cc"],
    hdrs = ["graph_collection_operator.h"],
    deps = [
        ":candidate_graph_runner",
        ":grow_stats",
        ":input_data",
        ":input_target",
        ":leaf_model_operators",
        ":params",
        ":split_collection_operators",
        "//tensorflow/contrib/tensor_forest:tree_utils",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
)

cc_library(
    name = "stat_utils",
    srcs = ["stat_utils.cc"],
    hdrs = ["stat_utils.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc",
        ],
        [
            "//third_party/eigen3",
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:fertile_stats_proto_cc_headers_only",
        ],
    ),
)

cc_library(
    name = "test_utils",
    hdrs = ["test_utils.h"],
    deps = [
        ":input_data",
        ":input_target",
    ],
)

cc_library(
    name = "params",
    srcs = ["params.cc"],
    hdrs = ["params.h"],
    deps = [
        "//third_party/eigen3",
        "//tensorflow/core:framework_headers_lib",
    ] + if_static(
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        ],
        [
            "//tensorflow/contrib/decision_trees/proto:generic_tree_model_cc_headers_only",
            "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc_headers_only",
        ],
    ),
)

tf_cc_test(
    name = "params_test",
    srcs = ["params_test.cc"],
    deps = [
        ":params",
        "//tensorflow/contrib/tensor_forest/proto:tensor_forest_params_proto_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
