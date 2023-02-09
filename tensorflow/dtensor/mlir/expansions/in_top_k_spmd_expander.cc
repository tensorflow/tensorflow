/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/in_top_k_spmd_expander.h"

#include <string>

#include "absl/types/optional.h"
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Creates a new layout for the "predictions" input based on the provided
// layout, ensuring that the 2nd dimension is replicated.
StatusOr<Layout> GetSuggestedPredictionsLayout(const Layout& layout) {
  // predictions is a rank-2 tensor (batch_size x num_classes)
  std::vector<ShardingSpec> layout_specs(2);
  layout_specs[0].set_sharding_spec(layout.sharding_spec(0));
  layout_specs[1].set_sharding_spec(Layout::kUnshardedDim);

  return Layout::GetLayout(layout_specs, layout.mesh());
}

// Creates a new layout based on the input "layout" but matching the batch dim
// of "other_layout".
StatusOr<Layout> MatchBatchDim(const Layout& layout,
                               const Layout& other_layout) {
  std::vector<ShardingSpec> layout_specs(layout.rank());
  layout_specs[0].set_sharding_spec(other_layout.sharding_spec(0));
  for (int i = 1; i < layout.rank(); ++i) {
    layout_specs[i].set_sharding_spec(layout.sharding_spec(i));
  }

  return Layout::GetLayout(layout_specs, layout.mesh());
}

}  // namespace

StatusOr<mlir::Operation*> InTopKSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto in_top_k_op = mlir::cast<mlir::TF::InTopKV2Op>(op);

  mlir::Value predictions = in_top_k_op.getPredictions();
  TF_ASSIGN_OR_RETURN(const Layout predictions_layout,
                      ExtractRequiredLayoutFromOperand(predictions));
  mlir::Value targets = in_top_k_op.getTargets();
  TF_ASSIGN_OR_RETURN(const Layout targets_layout,
                      ExtractRequiredLayoutFromOperand(targets));

  TF_ASSIGN_OR_RETURN(const Layout precision_layout,
                      ExtractRequiredSingleLayoutFromOp(in_top_k_op));

  bool relayout_predictions = false;
  bool relayout_targets = false;
  Layout new_predictions_layout = predictions_layout;
  Layout new_targets_layout = targets_layout;

  // Last dim (classes) of "predictions" is required to be replicated.
  if (!predictions_layout.IsLastDimReplicated()) {
    TF_ASSIGN_OR_RETURN(new_predictions_layout,
                        GetSuggestedPredictionsLayout(new_predictions_layout));
    relayout_predictions = true;
  }

  // If the batch dims of "targets" and "predictions" don't match, relayout the
  // less-sharded input to match the sharding of the other.
  const std::string& predictions_batch_dim =
      new_predictions_layout.sharding_spec(0);
  const std::string& targets_batch_dim = new_targets_layout.sharding_spec(0);
  if (predictions_batch_dim != targets_batch_dim) {
    if (Layout::IsShardedDimension(targets_batch_dim) &&
        Layout::IsShardedDimension(predictions_batch_dim)) {
      // Since "targets" is the smaller tensor, relayout it to match
      // "predictions".
      TF_ASSIGN_OR_RETURN(
          new_targets_layout,
          MatchBatchDim(new_targets_layout, new_predictions_layout));
      relayout_targets = true;
    } else if (Layout::IsShardedDimension(targets_batch_dim)) {
      TF_ASSIGN_OR_RETURN(
          new_predictions_layout,
          MatchBatchDim(new_predictions_layout, new_targets_layout));
      relayout_predictions = true;
    } else if (Layout::IsShardedDimension(predictions_batch_dim)) {
      TF_ASSIGN_OR_RETURN(
          new_targets_layout,
          MatchBatchDim(new_targets_layout, new_predictions_layout));
      relayout_targets = true;
    }
  }

  mlir::OpBuilder builder(op);
  mlir::IRMapping mapping;
  // Apply any input relayouts.
  if (relayout_predictions) {
    TF_ASSIGN_OR_RETURN(
        mlir::Value new_predictions,
        EmitRelayout(predictions, predictions_layout, new_predictions_layout));
    mapping.map(op->getOperand(0), new_predictions);
  }
  if (relayout_targets) {
    TF_ASSIGN_OR_RETURN(
        mlir::Value new_targets,
        EmitRelayout(targets, targets_layout, new_targets_layout));
    mapping.map(op->getOperand(1), new_targets);
  }

  mlir::Operation* new_op = builder.clone(*op, mapping);
  new_op = InferSPMDExpandedLocalShape(new_op);

  TF_ASSIGN_OR_RETURN(mlir::Value new_precision,
                      EmitRelayout(new_op->getOpResult(0), new_targets_layout,
                                   precision_layout));
  op->getResult(0).replaceAllUsesWith(new_precision);
  op->erase();

  return new_precision.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> InTopKSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // output is a rank-1 tensor (batch_size,)
  std::vector<std::string> layout_specs(1);
  layout_specs[0] = Layout::kUnshardedDim;

  // Set the output (precision) layout based on the inputs predictions and
  // targets. Prefer whichever is more sharded on the batch dimension.
  // Since "targets" is the smaller tensor, match its batch dimension first,
  // and override it with the batch dimension of "predictions" if that is
  // (possibly differently) sharded.
  if (input_layouts.find(1) != input_layouts.end()) {
    const Layout& targets_layout = input_layouts.lookup(1);
    const std::string& targets_batch_dim = targets_layout.sharding_spec(0);
    if (Layout::IsShardedDimension(targets_batch_dim)) {
      layout_specs[0] = targets_batch_dim;
    }
  }
  if (input_layouts.find(0) != input_layouts.end()) {
    const Layout& predictions_layout = input_layouts.lookup(0);
    const std::string& predictions_batch_dim =
        predictions_layout.sharding_spec(0);
    if (Layout::IsShardedDimension(predictions_batch_dim)) {
      layout_specs[0] = predictions_batch_dim;
    }
  }

  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      Layout::GetLayout(layout_specs, mesh));
  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> InTopKSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If no output layout is present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout& output_layout = output_layouts.lookup(0);

  // Set the layouts of predictions and targets based on the output. Last
  // dimension of predictions needs to be replicated.
  TF_ASSIGN_OR_RETURN(const Layout predictions_layout,
                      GetSuggestedPredictionsLayout(output_layout));
  const Layout targets_layout = output_layout;
  const Layout k_layout =
      Layout::ReplicatedOnMesh(output_layout.mesh(), /*rank=*/0);

  return llvm::DenseMap<int, Layout>(
      {{0, predictions_layout}, {1, targets_layout}, {2, k_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
