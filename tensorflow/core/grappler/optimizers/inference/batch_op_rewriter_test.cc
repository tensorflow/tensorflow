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
void AddBatchOp(GraphDef* graph, int num_batch_threads) {
  auto set_batch_node_attribute = [](const int32 num_batch_threads,
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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
