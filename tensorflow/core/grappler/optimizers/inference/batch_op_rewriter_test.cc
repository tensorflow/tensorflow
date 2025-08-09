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

#include <string_view>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "absl/container/flat_hash_map.h"
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
void AddBatchOp(GraphDef* graph, int num_batch_threads = 16,
                const absl::flat_hash_map<string, int>& reserved_int_attrs = {},
                int max_batch_size = 16, int batch_timeout_micros = 10000,
                const std::vector<int32>& allowed_batch_sizes = {8, 16},
                int max_enqueued_batches = 1000,
                bool disable_large_batch_splitting = false,
                std::string_view mixed_priority_policy = "",
                int low_priority_max_batch_size = -1,
                int low_priority_batch_timeout_micros = -1,
                const std::vector<int32>& low_priority_allowed_batch_sizes = {},
                int low_priority_max_enqueued_batches = -1) {
  auto set_batch_node_attribute = [&](const int32_t num_batch_threads,
                                      NodeDef* batch_op) {
    batch_op->set_name("cond/batch/BatchFunction");
    batch_op->set_op("BatchFunction");
    ::tensorflow::graph_transforms::SetNodeAttr("num_batch_threads",
                                                num_batch_threads, batch_op);
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
  const std::vector<int32> allowed_batch_sizes{4, 32};
  (*config.mutable_batch_options())["model_with_override"]
      .mutable_allowed_batch_sizes()
      ->Add(allowed_batch_sizes.begin(), allowed_batch_sizes.end());
  (*config.mutable_batch_options())["model_with_override"]
      .set_max_enqueued_batches(500);
  (*config.mutable_batch_options())["model_with_override"]
      .set_disable_large_batch_splitting(true);

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
  AddBatchOp(&expected_graph, 2 /* num_batch_threads */,
             {} /* reserved_int_attrs */, 128 /* max_batch_size */,
             5000 /* batch_timeout_micros */, allowed_batch_sizes,
             500 /* max_enqueued_batches */,
             true /* disable_large_batch_splitting */);

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
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
             /* low_priority_max_enqueued_batches = */ 1000);

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
