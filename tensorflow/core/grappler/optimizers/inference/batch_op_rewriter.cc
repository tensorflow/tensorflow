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

#include "google/protobuf/map.h"
#include "google/protobuf/repeated_field.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.h"
#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace grappler {

namespace {

constexpr char kBatchFunction[] = "BatchFunction";
constexpr char kBatchOpRewriteConfigParamKey[] = "batch_op_rewrite_config";
constexpr char kNumBatchThreadsAttr[] = "num_batch_threads";

}  // namespace
using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::Status;
using ::tensorflow::grappler::Cluster;
using ::tensorflow::grappler::GrapplerItem;

Status BatchOpRewriter::Init(
    const ::tensorflow::RewriterConfig_CustomGraphOptimizer* config) {
  // Parse the config from params. Fail if its missing or fails to parse.
  if (config->parameter_map().find(kBatchOpRewriteConfigParamKey) ==
      config->parameter_map().end()) {
    return ::tensorflow::errors::Internal(
        "batch_op_rewrite_config param must be set in the rewriter config "
        "with a serialized/encoded BatchOpRewriteConfig.");
  }
  const auto& params =
      config->parameter_map().at(kBatchOpRewriteConfigParamKey);
  std::string unencoded;
  if (params.s().empty()) {
    // If all parameters of BatchOpRewriteConfig have its default value
    // (e.g., enable_adaptive_shared_batching_thread_pool is false), proto
    // is considered as empty.
    VLOG(2) << "Empty batch-op rewrite config";
    return ::tensorflow::Status::OK();
  }
  if (!absl::Base64Unescape(params.s(), &unencoded)) {
    return ::tensorflow::errors::Internal(
        "Failed to unencode batch_op_rewrite_config from params.");
  }
  if (!config_.ParseFromString(unencoded)) {
    return ::tensorflow::errors::Internal(
        "Failed to parse batch_op_rewrite_config from params.");
  }
  VLOG(2) << "BatchOp Rewrite config is " << config_.DebugString();
  return ::tensorflow::Status::OK();
}

Status BatchOpRewriter::Optimize(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  VLOG(2) << "Running BatchOp Rewriter";
  *optimized_graph = item.graph;

  if (config_.enable_adaptive_shared_batching_thread_pool()) {
    // Go through all nodes and set 'num_batch_threads' to 0.
    // In for-loop here and below, use index (not range-based loop)to get
    // pointers (not reference) because helper function
    // `::tensorflow::graph_transforms::SetNodeAttr` doesn't have an override
    // that modifies reference.
    for (int i = 0; i < optimized_graph->node_size(); ++i) {
      NodeDef* node = optimized_graph->mutable_node(i);
      if ((node->op() == kBatchFunction) &&
          (node->attr().find(kNumBatchThreadsAttr) != node->attr().end())) {
        ::tensorflow::graph_transforms::SetNodeAttr(kNumBatchThreadsAttr, 0,
                                                    node);
      }
    }
    for (int i = 0; i < optimized_graph->library().function_size(); i++) {
      FunctionDef* function_def =
          optimized_graph->mutable_library()->mutable_function(i);
      for (int j = 0; j < function_def->node_def_size(); j++) {
        NodeDef* node = function_def->mutable_node_def(j);
        if ((node->op() == kBatchFunction) &&
            (node->attr().find(kNumBatchThreadsAttr) != node->attr().end())) {
          ::tensorflow::graph_transforms::SetNodeAttr(kNumBatchThreadsAttr, 0,
                                                      node);
        }
      }
    }
  }
  return Status::OK();
}

REGISTER_GRAPH_OPTIMIZER_AS(BatchOpRewriter, "batch_op_rewrite");

}  // namespace grappler
}  // namespace tensorflow
