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

load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

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

# Utils

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
        "//third_party/eigen3",
    ],
)

tf_cc_test(
    name = "sparse_column_iterable_test",
    size = "small",
    srcs = ["utils/sparse_column_iterable_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "examples_iterable_test",
    size = "small",
    srcs = ["utils/examples_iterable_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "example_test",
    size = "small",
    srcs = ["utils/example_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
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

tf_cc_test(
    name = "dropout_utils_test",
    size = "small",
    srcs = ["utils/dropout_utils_test.cc"],
    deps = [
        ":utils",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Models

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

tf_cc_test(
    name = "multiple_additive_trees_test",
    size = "small",
    srcs = ["models/multiple_additive_trees_test.cc"],
    deps = [
        ":batch_features_testutil",
        ":models",
        ":random_tree_gen",
        "//tensorflow/contrib/boosted_trees/resources:decision_tree_ensemble_resource",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Testutil

cc_library(
    name = "batch_features_testutil",
    testonly = 1,
    srcs = ["testutil/batch_features_testutil.cc"],
    hdrs = ["testutil/batch_features_testutil.h"],
    deps = [
        ":utils",
        "//tensorflow/core:framework_headers_lib",
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

tf_cc_binary(
    name = "random_tree_gen_main",
    srcs = ["testutil/random_tree_gen_main.cc"],
    deps = [
        ":random_tree_gen",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
    ],
)

# Quantiles

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

tf_cc_test(
    name = "weighted_quantiles_buffer_test",
    size = "small",
    srcs = ["quantiles/weighted_quantiles_buffer_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "weighted_quantiles_summary_test",
    size = "small",
    srcs = ["quantiles/weighted_quantiles_summary_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_cc_test(
    name = "weighted_quantiles_stream_test",
    size = "small",
    srcs = ["quantiles/weighted_quantiles_stream_test.cc"],
    deps = [
        ":weighted_quantiles",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Trees

cc_library(
    name = "trees",
    srcs = ["trees/decision_tree.cc"],
    hdrs = ["trees/decision_tree.h"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:framework_headers_lib",
    ],
)

tf_cc_test(
    name = "trees_test",
    size = "small",
    srcs = ["trees/decision_tree_test.cc"],
    deps = [
        ":trees",
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Learner/batch

py_library(
    name = "base_split_handler",
    srcs = ["learner/batch/base_split_handler.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/boosted_trees:batch_ops_utils_py",
        "//tensorflow/python:control_flow_ops",
    ],
)

py_library(
    name = "categorical_split_handler",
    srcs = ["learner/batch/categorical_split_handler.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_split_handler",
        "//tensorflow/contrib/boosted_trees:split_handler_ops_py",
        "//tensorflow/contrib/boosted_trees:stats_accumulator_ops_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
    ],
)

py_test(
    name = "categorical_split_handler_test",
    srcs = ["learner/batch/categorical_split_handler_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_split_handler",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "ordinal_split_handler",
    srcs = ["learner/batch/ordinal_split_handler.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":base_split_handler",
        "//tensorflow/contrib/boosted_trees:quantile_ops_py",
        "//tensorflow/contrib/boosted_trees:split_handler_ops_py",
        "//tensorflow/contrib/boosted_trees:stats_accumulator_ops_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:function",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
    ],
)

py_test(
    name = "ordinal_split_handler_test",
    srcs = ["learner/batch/ordinal_split_handler_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ordinal_split_handler",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_py",
        "//tensorflow/contrib/boosted_trees/proto:split_info_proto_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resources",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
    ],
)

# Learner/Common

cc_library(
    name = "class-partition-key",
    hdrs = ["learner/common/accumulators/class-partition-key.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_library(
    name = "feature-stats-accumulator",
    hdrs = ["learner/common/accumulators/feature-stats-accumulator.h"],
    deps = [
        ":class-partition-key",
    ],
)

tf_cc_test(
    name = "feature-stats-accumulator_test",
    size = "small",
    srcs = ["learner/common/accumulators/feature-stats-accumulator_test.cc"],
    deps = [
        ":feature-stats-accumulator",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "example_partitioner",
    srcs = ["learner/common/partitioners/example_partitioner.cc"],
    hdrs = ["learner/common/partitioners/example_partitioner.h"],
    deps = [
        "//tensorflow/contrib/boosted_trees/lib:trees",
        "//tensorflow/contrib/boosted_trees/lib:utils",
        "//tensorflow/core:framework_headers_lib",
    ],
)

tf_cc_test(
    name = "example_partitioner_test",
    size = "small",
    srcs = ["learner/common/partitioners/example_partitioner_test.cc"],
    deps = [
        ":example_partitioner",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Learner/stochastic
cc_library(
    name = "gradient-stats",
    hdrs = ["learner/stochastic/stats/gradient-stats.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "node-stats",
    hdrs = ["learner/stochastic/stats/node-stats.h"],
    deps = [
        ":gradient-stats",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
)

cc_library(
    name = "split-stats",
    hdrs = ["learner/stochastic/stats/split-stats.h"],
    deps = [
        ":node-stats",
    ],
)

cc_library(
    name = "feature-split-candidate",
    hdrs = ["learner/stochastic/stats/feature-split-candidate.h"],
    deps = [
        ":split-stats",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_cc",
    ],
)

tf_cc_test(
    name = "node-stats_test",
    size = "small",
    srcs = ["learner/stochastic/stats/node-stats_test.cc"],
    deps = [
        ":node-stats",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
