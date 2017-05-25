# Description:
#   This directory contains common utilities used in boosted_trees.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//tensorflow/contrib/boosted_trees:__subpackages__",
        "//tensorflow/contrib/boosted_trees:friends",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

cc_library(
    name = "weighted_quantiles",
    srcs = [],
    hdrs = [
        "quantiles/weighted_quantiles_buffer.h",
        "quantiles/weighted_quantiles_stream.h",
        "quantiles/weighted_quantiles_summary.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "weighted_quantiles_buffer_test",
    size = "small",
    srcs = ["quantiles/weighted_quantiles_buffer_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "weighted_quantiles_summary_test",
    size = "small",
    srcs = ["quantiles/weighted_quantiles_summary_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "weighted_quantiles_stream_test",
    size = "small",
    srcs = ["quantiles/weighted_quantiles_stream_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "utils",
    srcs = [
        "utils/batch_features.cc",
        "utils/dropout_utils.cc",
        "utils/examples_iterable.cc",
        "utils/parallel_for.cc",
        "utils/sparse_column_iterable.cc",
        "utils/tensor_utils.cc",
    ],
    hdrs = [
        "utils/batch_features.h",
        "utils/dropout_utils.h",
        "utils/example.h",
        "utils/examples_iterable.h",
        "utils/macros.h",
        "utils/optional_value.h",
        "utils/parallel_for.h",
        "utils/random.h",
        "utils/sparse_column_iterable.h",
        "utils/tensor_utils.h",
    ],
    deps = [
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//third_party/eigen3",
    ],
)

cc_test(
    name = "sparse_column_iterable_test",
    size = "small",
    srcs = ["utils/sparse_column_iterable_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "examples_iterable_test",
    size = "small",
    srcs = ["utils/examples_iterable_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "batch_features_test",
    size = "small",
    srcs = ["utils/batch_features_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_test(
    name = "dropout_utils_test",
    size = "small",
    srcs = ["utils/dropout_utils_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "models",
    srcs = ["models/multiple_additive_trees.cc"],
    hdrs = ["models/multiple_additive_trees.h"],
    deps = [
        ":trees",
        ":utils",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "multiple_additive_trees_test",
    size = "small",
    srcs = ["models/multiple_additive_trees_test.cc"],
    deps = [
        ":batch_features_testutil",
        ":models",
        ":random_tree_gen",
        "//tensorflow/contrib/boosted_trees/resources:decision_tree_ensemble_resource",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "trees",
    srcs = ["trees/decision_tree.cc"],
    hdrs = ["trees/decision_tree.h"],
    deps = [
        ":utils",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "trees_test",
    size = "small",
    srcs = ["trees/decision_tree_test.cc"],
    deps = [
        ":trees",
        ":utils",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "batch_features_testutil",
    testonly = 1,
    srcs = ["testutil/batch_features_testutil.cc"],
    hdrs = ["testutil/batch_features_testutil.h"],
    deps = [
        ":utils",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "random_tree_gen",
    srcs = ["testutil/random_tree_gen.cc"],
    hdrs = ["testutil/random_tree_gen.h"],
    deps = [
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:lib",
    ],
)

cc_binary(
    name = "random_tree_gen_main",
    srcs = ["testutil/random_tree_gen_main.cc"],
    deps = [
        ":random_tree_gen",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)
