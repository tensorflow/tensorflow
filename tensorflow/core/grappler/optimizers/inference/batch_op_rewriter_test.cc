/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.h"

#include <string>
#include <string_view>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/escaping.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::RewriterConfig_CustomGraphOptimizer;
using ::tensorflow::Status;
using ::tensorflow::grappler::GrapplerItem;
using ::tensorflow::serving::BatchOpRewriteConfig;

// Add batch op in both GraphDef.node and GraphDef.library.function.node_def.
void AddBatchOp(
    GraphDef* graph, int num_batch_threads = 16,
    const absl::flat_hash_map<std::string, int>& reserved_int_attrs = {},
    int max_batch_size = 16, int batch_timeout_micros = 10000,
    const std::vector<int32_t>& allowed_batch_sizes = {8, 16},
    int max_enqueued_batches = 1000, bool disable_large_batch_splitting = false,
    std::string_view mixed_priority_policy = "",
    int low_priority_max_batch_size = -1,
    int low_priority_batch_timeout_micros = -1,
    const std::vector<int32_t>& low_priority_allowed_batch_sizes = {},
    int low_priority_max_enqueued_batches = -1,
    int num_warmup_batch_threads = 0) {
  auto set_batch_node_attribute = [&](const int32_t num_batch_threads,
                                      NodeDef* batch_op) {
    batch_op->set_name("cond/batch/BatchFunction");
    batch_op->set_op("BatchFunction");
    ::tensorflow::graph_transforms::SetNodeAttr("num_batch_threads",
                                                num_batch_threads, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr(
        "num_warmup_batch_threads", num_warmup_batch_threads, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("max_batch_size",
                                                max_batch_size, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("batch_timeout_micros",
                                                batch_timeout_micros, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("allowed_batch_sizes",
                                                allowed_batch_sizes, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("max_enqueued_batches",
                                                max_enqueued_batches, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("enable_large_batch_splitting",
                                                !disable_large_batch_splitting,
                                                batch_op);

    if (!mixed_priority_policy.empty()) {
      ::tensorflow::graph_transforms::SetNodeAttr(
          "mixed_priority_policy", mixed_priority_policy, batch_op);
    }
    if (low_priority_max_batch_size != -1) {
      ::tensorflow::graph_transforms::SetNodeAttr(
          "low_priority_max_batch_size", low_priority_max_batch_size, batch_op);
    }
    if (low_priority_batch_timeout_micros != -1) {
      ::tensorflow::graph_transforms::SetNodeAttr(
          "low_priority_batch_timeout_micros",
          low_priority_batch_timeout_micros, batch_op);
    }
    if (!low_priority_allowed_batch_sizes.empty()) {
      ::tensorflow::graph_transforms::SetNodeAttr(
          "low_priority_allowed_batch_sizes", low_priority_allowed_batch_sizes,
          batch_op);
    }
    if (low_priority_max_enqueued_batches != -1) {
      ::tensorflow::graph_transforms::SetNodeAttr(
          "low_priority_max_enqueued_batches",
          low_priority_max_enqueued_batches, batch_op);
    }

    if (!reserved_int_attrs.empty()) {
      ::tensorflow::graph_transforms::SetNodeAttr(kEnableAdaptiveSchedulerAttr,
                                                  true, batch_op);
      for (const auto& reserved_int_attr : reserved_int_attrs) {
        ::tensorflow::graph_transforms::SetNodeAttr(
            reserved_int_attr.first, reserved_int_attr.second, batch_op);
      }
    }
  };

  // Add batch op in GraphDef.node.
  set_batch_node_attribute(num_batch_threads, graph->add_node());

  // Add batch op in GraphDef.library.function.node_def.
  FunctionDefLibrary* function_def_lib = graph->mutable_library();
  FunctionDef* function_def = function_def_lib->add_function();
  set_batch_node_attribute(num_batch_threads, function_def->add_node_def());
}

RewriterConfig_CustomGraphOptimizer MakeConfig(
    const BatchOpRewriteConfig& config) {
  RewriterConfig_CustomGraphOptimizer rewriter_config;
  (*rewriter_config.mutable_parameter_map())["batch_op_rewrite_config"].set_s(
      absl::Base64Escape(config.SerializeAsString()));
  return rewriter_config;
}

// Returns a graph with the batch op that AddBatchOp() adds by default, except
// for its `num_batch_threads`.
GraphDef GraphWithBatchOp(int num_batch_threads = 16) {
  GraphDef graph;
  AddBatchOp(&graph, num_batch_threads);
  return graph;
}

// Runs the rewriter with `config` on GraphWithBatchOp(), in a session for the
// model named `model_name`. Checks that Init() succeeds, and returns the status
// of Optimize().
Status RewriteGraphWithBatchOp(const BatchOpRewriteConfig& config,
                               const std::string& model_name,
                               GraphDef* optimized_graph) {
  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      model_name);
  BatchOpRewriter optimizer;
  TF_CHECK_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));
  GrapplerItem item;
  item.graph = GraphWithBatchOp();
  return optimizer.Optimize(nullptr, item, optimized_graph);
}

class BatchOpRewriterTest : public ::testing::TestWithParam<bool> {};

INSTANTIATE_TEST_SUITE_P(RewriteNumBatchThreads, BatchOpRewriterTest,
                         ::testing::Bool());

TEST_P(BatchOpRewriterTest, Basic) {
  GrapplerItem item;
  AddBatchOp(&item.graph, 16);
  BatchOpRewriteConfig config;
  config.set_enable_adaptive_shared_batching_thread_pool(GetParam());
  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.Init(&rewriter_config));
  GraphDef optimized_graph;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));
  // We can't use the testing::EqualsProto matcher because it is not available
  // in OSS.
  GraphDef expected_graph;
  AddBatchOp(&expected_graph, GetParam() ? 0 : 16);

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

// Tests that invalid argument are caught.
TEST_P(BatchOpRewriterTest, InvalidArgumentForAdaptiveBatchScheduler) {
  GrapplerItem item;
  AddBatchOp(&item.graph, 16);
  BatchOpRewriteConfig config;

  // PARSE_TEXT_PROTO isn't available in TF OSS.
  config.set_enable_adaptive_shared_batching_thread_pool(GetParam());
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_batches_to_average_over()
      ->set_value(1000);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_initial_inflight_batches_limit()
      ->set_value(8);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_min_inflight_batches_limit()
      ->set_value(16);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_max_inflight_batches_limit()
      ->set_value(32);

  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.Init(&rewriter_config));
  optimizer.config_proto_.mutable_experimental()
      ->mutable_session_metadata()
      ->set_version(123);
  optimizer.config_proto_.mutable_experimental()
      ->mutable_session_metadata()
      ->set_name("model_with_override");
  GraphDef optimized_graph;
  Status status = optimizer.Optimize(nullptr, item, &optimized_graph);

  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::IsInvalidArgument(status));
}

// Tests that reserved attributes relevant with adaptive scheduler are
// overridden in the output GraphDef.
TEST_P(BatchOpRewriterTest, AdaptiveBatchScheduler) {
  BatchOpRewriteConfig config;

  // PARSE_TEXT_PROTO isn't available in TF OSS.
  config.set_enable_adaptive_shared_batching_thread_pool(GetParam());
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_adaptive_batch_scheduler_option()
      ->mutable_batches_to_average_over()
      ->set_value(1000);
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_adaptive_batch_scheduler_option()
      ->mutable_initial_inflight_batches_limit()
      ->set_value(16);
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_adaptive_batch_scheduler_option()
      ->mutable_min_inflight_batches_limit()
      ->set_value(8);
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_adaptive_batch_scheduler_option()
      ->mutable_max_inflight_batches_limit()
      ->set_value(32);
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_adaptive_batch_scheduler_option()
      ->mutable_full_batch_scheduling_boost_micros()
      ->set_value(12345);

  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  config_proto.mutable_experimental()->mutable_session_metadata()->set_version(
      123);
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      "model_with_override");
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));

  GraphDef optimized_graph;
  GrapplerItem item;
  AddBatchOp(&item.graph, 16);
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));
  // We can't use the testing::EqualsProto matcher because it is not available
  // in OSS.
  GraphDef expected_graph;
  // For `model_with_override`, attribute `num_batch_threads` is 16 and not
  // overridden to zero regardless of
  // `enable_adaptive_shared_batching_thread_pool`, since `model_with_override`
  // override its scheduler options in `model_scheduler_options`.
  AddBatchOp(&expected_graph, 16 /* num_batch_threads */,
             {{kBatchesToAverageOverAttr, 1000},
              {kInitialInflightBatchesAttr, 16},
              {kMinInflightBatchesAttr, 8},
              {kMaxInflightBatchesAttr, 32},
              {kFullBatchSchedulingBoostMicros, 12345}});

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

// Tests that using the deprecated model_scheduler_options returns an error.
TEST_F(BatchOpRewriterTest, UpdateModelSchedulerOptions) {
  BatchOpRewriteConfig config;

  // PARSE_TEXT_PROTO isn't available in TF OSS.
  config.set_enable_adaptive_shared_batching_thread_pool(true);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_batches_to_average_over()
      ->set_value(1000);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_initial_inflight_batches_limit()
      ->set_value(16);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_min_inflight_batches_limit()
      ->set_value(8);
  (*config.mutable_model_scheduler_options())["model_with_override"]
      .mutable_max_inflight_batches_limit()
      ->set_value(32);

  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  config_proto.mutable_experimental()->mutable_session_metadata()->set_version(
      123);
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      "model_with_override");
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));
  GraphDef optimized_graph;
  GrapplerItem item;
  AddBatchOp(&item.graph, 16);
  ASSERT_FALSE(optimizer.Optimize(nullptr, item, &optimized_graph).ok());
}

// Tests that the updated batch options are present in the output GraphDef.
TEST_F(BatchOpRewriterTest, UpdateBatchOptions) {
  BatchOpRewriteConfig config;

  // PARSE_TEXT_PROTO isn't available in TF OSS.
  (*config.mutable_batch_options())["model_with_override"]
      .set_num_batch_threads(2);
  (*config.mutable_batch_options())["model_with_override"].set_max_batch_size(
      128);
  (*config.mutable_batch_options())["model_with_override"]
      .set_batch_timeout_micros(5000);
  const std::vector<int32_t> allowed_batch_sizes{4, 32};
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_allowed_batch_sizes()
      ->Add(allowed_batch_sizes.begin(), allowed_batch_sizes.end());
  (*config.mutable_batch_options())["model_with_override"]
      .set_max_enqueued_batches(500);
  (*config.mutable_batch_options())["model_with_override"]
      .set_disable_large_batch_splitting(true);
  (*config.mutable_batch_options())["model_with_override"]
      .set_num_warmup_batch_threads(5);

  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  config_proto.mutable_experimental()->mutable_session_metadata()->set_version(
      123);
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      "model_with_override");
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));

  GraphDef optimized_graph;
  GrapplerItem item;
  AddBatchOp(&item.graph);
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));
  // We can't use the testing::EqualsProto matcher because it is not available
  // in OSS.
  GraphDef expected_graph;
  // For `model_with_override`, attribute `num_batch_threads` is 16 and not
  // overridden to zero regardless of
  // `enable_adaptive_shared_batching_thread_pool`, since `model_with_override`
  // override its scheduler options in `model_scheduler_options`.
  AddBatchOp(&expected_graph, /* num_batch_threads = */ 2,
             /* reserved_int_attrs = */ {},
             /* max_batch_size = */ 128,
             /* batch_timeout_micros = */ 5000,
             /* allowed_batch_sizes = */ allowed_batch_sizes,
             /* max_enqueued_batches = */ 500,
             /* disable_large_batch_splitting = */ true,
             /* mixed_priority_policy = */ "",
             /* low_priority_max_batch_size = */ -1,
             /* low_priority_batch_timeout_micros = */ -1,
             /* low_priority_allowed_batch_sizes = */ {},
             /* low_priority_max_enqueued_batches = */ -1,
             /* num_warmup_batch_threads = */ 5);

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

TEST_F(BatchOpRewriterTest, UpdateGlobalBatchOptions) {
  BatchOpRewriteConfig config;
  config.mutable_global_batch_options()->set_num_batch_threads(4);
  config.mutable_global_batch_options()->set_max_batch_size(64);

  // Specific model override.
  (*config.mutable_batch_options())["model_with_override"]
      .set_num_batch_threads(2);

  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  config_proto.mutable_experimental()->mutable_session_metadata()->set_version(
      123);

  // Test model WITH override.
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      "model_with_override");
  BatchOpRewriter optimizer_with_override;
  TF_ASSERT_OK(
      optimizer_with_override.InitWithConfig(config_proto, &rewriter_config));

  GraphDef optimized_graph_override;
  GrapplerItem item_override;
  AddBatchOp(&item_override.graph, /* num_batch_threads = */ 16);
  TF_ASSERT_OK(optimizer_with_override.Optimize(nullptr, item_override,
                                                &optimized_graph_override));

  GraphDef expected_graph_override;
  AddBatchOp(&expected_graph_override, /* num_batch_threads = */ 2);
  EXPECT_EQ(optimized_graph_override.DebugString(),
            expected_graph_override.DebugString());

  // Test model WITHOUT override (uses global).
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      "another_model");
  BatchOpRewriter optimizer_without_override;
  TF_ASSERT_OK(optimizer_without_override.InitWithConfig(config_proto,
                                                         &rewriter_config));

  GraphDef optimized_graph_global;
  GrapplerItem item_global;
  AddBatchOp(&item_global.graph);
  TF_ASSERT_OK(optimizer_without_override.Optimize(nullptr, item_global,
                                                   &optimized_graph_global));

  GraphDef expected_graph_global;
  AddBatchOp(&expected_graph_global, /* num_batch_threads = */ 4,
             /* reserved_int_attrs = */ {}, /* max_batch_size = */ 64);
  EXPECT_EQ(optimized_graph_global.DebugString(),
            expected_graph_global.DebugString());
}

// Tests that `regex_batch_options` can give options to a model whose name
// isn't fully known when the config is written.
TEST_F(BatchOpRewriterTest, UpdateRegexBatchOptions) {
  BatchOpRewriteConfig config;
  BatchOpRewriteConfig::RegexBatchOptions* regex_batch_options =
      config.add_regex_batch_options();
  regex_batch_options->set_model_name_regex("my_model__.*");
  regex_batch_options->mutable_batch_options()->set_num_batch_threads(2);
  regex_batch_options->mutable_batch_options()->set_max_batch_size(128);

  GraphDef optimized_graph;
  TF_ASSERT_OK(
      RewriteGraphWithBatchOp(config, "my_model__1b9d6bcd", &optimized_graph));

  GraphDef expected_graph;
  AddBatchOp(&expected_graph, /* num_batch_threads = */ 2,
             /* reserved_int_attrs = */ {}, /* max_batch_size = */ 128);
  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

// Tests that a regular expression has to match the whole model name.
TEST_F(BatchOpRewriterTest, RegexBatchOptionsMatchWholeModelName) {
  BatchOpRewriteConfig config;
  BatchOpRewriteConfig::RegexBatchOptions* regex_batch_options =
      config.add_regex_batch_options();
  regex_batch_options->set_model_name_regex("my_model");
  regex_batch_options->mutable_batch_options()->set_num_batch_threads(2);

  GraphDef optimized_graph;
  TF_ASSERT_OK(RewriteGraphWithBatchOp(config, "my_model", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp(2).DebugString());

  // The regular expression matches only a part of these names, so the graph
  // isn't changed.
  TF_ASSERT_OK(
      RewriteGraphWithBatchOp(config, "my_model__1b9d6bcd", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp().DebugString());
  TF_ASSERT_OK(
      RewriteGraphWithBatchOp(config, "not_my_model", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp().DebugString());
}

// Tests the order in which the options for a model are looked up: the entry of
// `batch_options` for its name, then the first entry of `regex_batch_options`
// that matches its name, then `global_batch_options`.
TEST_F(BatchOpRewriterTest, BatchOptionsPrecedence) {
  BatchOpRewriteConfig config;
  (*config.mutable_batch_options())["model_a"].set_num_batch_threads(1);
  BatchOpRewriteConfig::RegexBatchOptions* first_regex_batch_options =
      config.add_regex_batch_options();
  first_regex_batch_options->set_model_name_regex("model_.*");
  first_regex_batch_options->mutable_batch_options()->set_num_batch_threads(2);
  BatchOpRewriteConfig::RegexBatchOptions* second_regex_batch_options =
      config.add_regex_batch_options();
  second_regex_batch_options->set_model_name_regex(".*_b");
  second_regex_batch_options->mutable_batch_options()->set_num_batch_threads(3);
  config.mutable_global_batch_options()->set_num_batch_threads(4);

  GraphDef optimized_graph;
  // "model_a" has an entry in `batch_options`, which is used although the first
  // regular expression also matches.
  TF_ASSERT_OK(RewriteGraphWithBatchOp(config, "model_a", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp(1).DebugString());

  // Both regular expressions match "model_b", so the first one is used.
  TF_ASSERT_OK(RewriteGraphWithBatchOp(config, "model_b", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp(2).DebugString());

  // Only the second regular expression matches "another_b".
  TF_ASSERT_OK(RewriteGraphWithBatchOp(config, "another_b", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp(3).DebugString());

  // Nothing matches "another_c", so the global options are used.
  TF_ASSERT_OK(RewriteGraphWithBatchOp(config, "another_c", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp(4).DebugString());
}

// Tests that the keys of `batch_options` are model names and not regular
// expressions.
TEST_F(BatchOpRewriterTest, BatchOptionsKeyIsNotRegex) {
  BatchOpRewriteConfig config;
  (*config.mutable_batch_options())["model_.*"].set_num_batch_threads(2);

  GraphDef optimized_graph;
  TF_ASSERT_OK(RewriteGraphWithBatchOp(config, "model_a", &optimized_graph));
  EXPECT_EQ(optimized_graph.DebugString(), GraphWithBatchOp().DebugString());
}

// Tests that an invalid regular expression is an error for every model, and
// that Optimize() returns it. If Init() returned it, the meta optimizer
// wouldn't run any of its optimizers.
TEST_F(BatchOpRewriterTest, InvalidRegexBatchOptions) {
  BatchOpRewriteConfig config;
  (*config.mutable_batch_options())["model_a"].set_num_batch_threads(2);
  BatchOpRewriteConfig::RegexBatchOptions* regex_batch_options =
      config.add_regex_batch_options();
  regex_batch_options->set_model_name_regex("model_(");
  regex_batch_options->mutable_batch_options()->set_num_batch_threads(3);

  GraphDef optimized_graph;
  // "model_a" has an entry in `batch_options`, and "model_b" doesn't.
  EXPECT_THAT(RewriteGraphWithBatchOp(config, "model_a", &optimized_graph),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("'model_('")));
  EXPECT_THAT(RewriteGraphWithBatchOp(config, "model_b", &optimized_graph),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     ::testing::HasSubstr("'model_('")));
}

// Tests that an empty regular expression, which is what an entry of
// `regex_batch_options` has when `model_name_regex` isn't set, is an error.
TEST_F(BatchOpRewriterTest, EmptyRegexBatchOptions) {
  BatchOpRewriteConfig config;
  config.add_regex_batch_options()
      ->mutable_batch_options()
      ->set_num_batch_threads(2);

  GraphDef optimized_graph;
  EXPECT_THAT(RewriteGraphWithBatchOp(config, "model_a", &optimized_graph),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  ::testing::HasSubstr("model_name_regex is empty")));
}

// Tests that an empty or invalid regular expression fails Optimize() and
// applies none of the options in the config even when `session_metadata` is
// unset on `ConfigProto`.
TEST_F(BatchOpRewriterTest, InvalidRegexBatchOptionsWithoutSessionMetadata) {
  GrapplerItem item;
  item.graph = GraphWithBatchOp(/*num_batch_threads=*/16);

  for (const char* bad_regex : {"", "model_("}) {
    BatchOpRewriteConfig config;
    config.set_enable_adaptive_shared_batching_thread_pool(true);
    config.mutable_global_prioritization()->set_num_threads(4);
    BatchOpRewriteConfig::RegexBatchOptions* regex_batch_options =
        config.add_regex_batch_options();
    regex_batch_options->set_model_name_regex(bad_regex);
    regex_batch_options->mutable_batch_options()->set_num_batch_threads(2);

    RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
    ConfigProto config_proto;
    BatchOpRewriter optimizer;
    TF_ASSERT_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));

    GraphDef optimized_graph;
    EXPECT_THAT(
        optimizer.Optimize(nullptr, item, &optimized_graph),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("model_name_regex")));
    EXPECT_EQ(optimized_graph.DebugString(),
              GraphWithBatchOp(/*num_batch_threads=*/16).DebugString());
  }
}

TEST_F(BatchOpRewriterTest,
       UpdateAdaptiveSharedBatchSchedulerAndNumBatchThreads) {
  GrapplerItem item;
  AddBatchOp(&item.graph, 16);
  BatchOpRewriteConfig config;
  config.set_enable_adaptive_shared_batching_thread_pool(true);
  (*config.mutable_batch_options())["model_with_override"]
      .set_num_batch_threads(2);
  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  config_proto.mutable_experimental()->mutable_session_metadata()->set_version(
      123);
  config_proto.mutable_experimental()->mutable_session_metadata()->set_name(
      "model_with_override");
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));
  GraphDef optimized_graph;
  ASSERT_FALSE(optimizer.Optimize(nullptr, item, &optimized_graph).ok());
}

TEST_F(BatchOpRewriterTest, UpdateToUseGlobalPrioritization) {
  GrapplerItem item;
  AddBatchOp(&item.graph, /* num_batch_threads = */ 16,
             /* reserved_int_attrs = */ {}, /* max_batch_size = */ 16,
             /* batch_timeout_micros = */ 10000,
             /* allowed_batch_sizes = */ {8, 16},
             /* max_enqueued_batches = */ 1000,
             /* disable_large_batch_splitting = */ false,
             /* mixed_priority_policy = */ "",
             /* low_priority_max_batch_size = */ -1,
             /* low_priority_batch_timeout_micros = */ 20000,
             /* low_priority_allowed_batch_sizes = */ {},
             /* low_priority_max_enqueued_batches = */ -1);

  BatchOpRewriteConfig config;
  config.mutable_global_prioritization()->set_num_threads(4);

  RewriterConfig_CustomGraphOptimizer rewriter_config = MakeConfig(config);
  ConfigProto config_proto;
  BatchOpRewriter optimizer;
  TF_ASSERT_OK(optimizer.InitWithConfig(config_proto, &rewriter_config));

  GraphDef optimized_graph;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));

  // NOTE: 'low_priority_batch_timeout_micros' keeps the same value as the
  // original graph since it had a value explicitly specified.
  GraphDef expected_graph;
  AddBatchOp(&expected_graph, /* num_batch_threads = */ 4,
             /* reserved_int_attrs = */ {}, /* max_batch_size = */ 16,
             /* batch_timeout_micros = */ 10000,
             /* allowed_batch_sizes = */ {8, 16},
             /* max_enqueued_batches = */ 1000,
             /* disable_large_batch_splitting = */ false,
             /* mixed_priority_policy = */ "priority_merge",
             /* low_priority_max_batch_size = */ 16,
             /* low_priority_batch_timeout_micros = */ 20000,
             /* low_priority_allowed_batch_sizes = */ {8, 16},
             /* low_priority_max_enqueued_batches = */ 1000,
             /* num_warmup_batch_threads = */ 0);

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
