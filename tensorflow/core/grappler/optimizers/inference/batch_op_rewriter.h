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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_INFERENCE_BATCH_OP_REWRITER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_INFERENCE_BATCH_OP_REWRITER_H_

#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/optimizers/inference/batch_op_rewriter.pb.h"

namespace tensorflow {
namespace grappler {

constexpr char kEnableAdaptiveSchedulerAttr[] = "_enable_adaptive_scheduler";
constexpr char kMinInflightBatchesAttr[] = "_min_inflight_batches";
constexpr char kInitialInflightBatchesAttr[] = "_initial_inflight_batches";
constexpr char kMaxInflightBatchesAttr[] = "_max_inflight_batches";
constexpr char kBatchesToAverageOverAttr[] = "_batches_to_average_over";

constexpr int64 kMinInflightBatches = 16;
constexpr int64 kInitialInflightBatches = 16;
constexpr int64 kBatchesToAverageOver = 10;
constexpr int64 kMaxInflightBatches = 64;

using ::tensorflow::serving::BatchOpRewriteConfig;

// This optimization does the following:
//
// Rewrite `num_batch_threads` to zero in batch-op. In this way, graphs with
// batch op will use a shared thread pool to schedule batches, as opposed to
// allocating batch threads per batch-op.
class BatchOpRewriter : public ::tensorflow::grappler::CustomGraphOptimizer {
 public:
  ::tensorflow::Status Init(
      const ::tensorflow::RewriterConfig_CustomGraphOptimizer* config) override;

  std::string name() const override { return "batch_op_rewriter"; }

  bool UsesFunctionLibrary() const override { return false; }

  ::tensorflow::Status Optimize(
      ::tensorflow::grappler::Cluster* cluster,
      const ::tensorflow::grappler::GrapplerItem& item,
      ::tensorflow::GraphDef* optimized_graph) override;

 private:
  BatchOpRewriteConfig config_;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_INFERENCE_BATCH_OP_REWRITER_H_
