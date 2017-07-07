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

cc_test(
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

cc_test(
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

# Testutil

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

cc_test(
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

cc_test(
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

cc_test(
    name = "example_partitioner_test",
    size = "small",
    srcs = ["learner/common/partitioners/example_partitioner_test.cc"],
    deps = [
        ":example_partitioner",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

# Learner/stochastic

cc_library(
    name = "feature-column-handlers",
    srcs = [
        "learner/stochastic/handlers/bias-feature-column-handler.cc",
        "learner/stochastic/handlers/categorical-feature-column-handler.cc",
        "learner/stochastic/handlers/dense-quantized-feature-column-handler.cc",
        "learner/stochastic/handlers/sparse-quantized-feature-column-handler.cc",
    ],
    hdrs = [
        "learner/stochastic/handlers/bias-feature-column-handler.h",
        "learner/stochastic/handlers/categorical-feature-column-handler.h",
        "learner/stochastic/handlers/dense-quantized-feature-column-handler.h",
        "learner/stochastic/handlers/feature-column-handler.h",
        "learner/stochastic/handlers/sparse-quantized-feature-column-handler.h",
    ],
    deps = [
        ":feature-split-candidate",
        ":feature-stats-accumulator",
        "//tensorflow/contrib/boosted_trees/proto:learner_proto_cc",
        "//tensorflow/core:framework_headers_lib",
    ],
)

cc_test(
    name = "feature-column-handlers_test",
    size = "small",
    srcs = [
        "learner/stochastic/handlers/bias-feature-column-handler_test.cc",
        "learner/stochastic/handlers/categorical-feature-column-handler_test.cc",
        "learner/stochastic/handlers/dense-quantized-feature-column-handler_test.cc",
        "learner/stochastic/handlers/sparse-quantized-feature-column-handler_test.cc",
    ],
    deps = [
        ":feature-column-handlers",
        "//tensorflow/core:lib",
        "//tensorflow/core:tensor_testutil",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

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

cc_test(
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
