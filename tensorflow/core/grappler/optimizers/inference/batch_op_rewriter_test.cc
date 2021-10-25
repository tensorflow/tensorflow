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
    GraphDef* graph, int num_batch_threads,
    const absl::flat_hash_map<string, int>& reserved_int_attrs = {}) {
  auto set_batch_node_attribute = [&](const int32_t num_batch_threads,
                                      NodeDef* batch_op) {
    batch_op->set_name("cond/batch/BatchFunction");
    batch_op->set_op("BatchFunction");
    ::tensorflow::graph_transforms::SetNodeAttr("num_batch_threads",
                                                num_batch_threads, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("max_batch_size", 16, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("batch_timeout_micros", 10000,
                                                batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr(
        "allowed_batch_sizes", std::vector<int32>{8, 16}, batch_op);
    ::tensorflow::graph_transforms::SetNodeAttr("max_enqueued_batches", 1000,
                                                batch_op);

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
  EXPECT_TRUE(errors::IsInvalidArgument(status));
}

// Tests that reserved attributes relevant with adaptive scheduler are
// overridden in the output GraphDef.
TEST_P(BatchOpRewriterTest, AdaptiveBatchScheduler) {
  BatchOpRewriteConfig config;

  // PARSE_TEXT_PROTO isn't available in TF OSS.
  config.set_enable_adaptive_shared_batching_thread_pool(GetParam());
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
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized_graph));
  // We can't use the testing::EqualsProto matcher because it is not available
  // in OSS.
  GraphDef expected_graph;
  // For `model_with_override`, attribute `num_batch_threads` is 16 and not
  // overridden to zero regardless of
  // `enable_adaptive_shared_batching_thread_pool`, since `model_with_override`
  // override its scheduler options in `model_scheduler_options`.
  AddBatchOp(&expected_graph, 16 /* num_batch_threads */,
             {
                 {kBatchesToAverageOverAttr, 1000},
                 {kInitialInflightBatchesAttr, 16},
                 {kMinInflightBatchesAttr, 8},
                 {kMaxInflightBatchesAttr, 32},
             });

  EXPECT_EQ(optimized_graph.DebugString(), expected_graph.DebugString());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
