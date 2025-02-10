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

#include "tensorflow/dtensor/mlir/expansions/dataparallel_spmd_expander.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Check all tensors are batch parallel
bool AllBatchParallel(const std::vector<Layout>& layouts,
                      const llvm::DenseMap<int, int>& batchable_indices) {
  for (int i = 0; i < layouts.size(); ++i) {
    if (!layouts[i].IsBatchParallel(batchable_indices.lookup(i))) return false;
  }
  return true;
}
// Check all Layouts have the same batch rank
bool SameBatchRank(const std::vector<Layout>& layouts,
                   const llvm::DenseMap<int, int>& batchable_indices) {
  absl::flat_hash_set<int> batch_ranks;
  // Add all batch ranks of layouts
  for (auto const& idx_and_non_batch_rank : batchable_indices) {
    auto const& idx = idx_and_non_batch_rank.first;
    auto const& non_batch_rank = idx_and_non_batch_rank.second;
    batch_ranks.insert(layouts[idx].rank() - non_batch_rank);
  }
  return batch_ranks.size() <= 1;
}

// Check if any layout from a set of indices is not a nullopt
bool AnyLayoutExist(const llvm::DenseMap<int, Layout>& layouts,
                    const llvm::DenseMap<int, int>& indices) {
  for (auto const& idx_and_unused : indices) {
    auto const& idx = idx_and_unused.first;
    if (layouts.find(idx) != layouts.end()) return true;
  }
  return false;
}
// Given layouts to merge and a map of {indices of the batchable layouts, rank
// of non batch dimensions} merges those batchable layouts to produce one single
// layout. Assumes all batch ranks are the same for all batchable layouts, which
// is enforced before this
//
// Merged together so that we have the layout which is sharded in a tensor dim
// if and only if all layouts are sharded in the same sharding_spec.
StatusOr<Layout> MergeBatchLayouts(
    const llvm::DenseMap<int, Layout>& layouts,
    const llvm::DenseMap<int, int>& batchable_args, const Mesh& mesh) {
  // Get the batch rank
  int layout_idx = -1;
  for (auto const& idx_and_unused : batchable_args) {
    auto const& idx = idx_and_unused.first;
    if (layouts.find(idx) != layouts.end()) layout_idx = idx;
  }

  int batch_rank =
      layouts.lookup(layout_idx).rank() - batchable_args.lookup(layout_idx);
  // Initialize with replicated
  std::vector<std::string> merged_specs(batch_rank, Layout::kUnshardedDim);

  // Merge layouts. If any dimension don't agree on sharding dim, then replicate
  for (int i = 0; i < batch_rank; ++i) {
    absl::flat_hash_set<std::string> spec_set;
    for (auto const& arg_idx_and_unused : batchable_args) {
      auto const& arg_idx = arg_idx_and_unused.first;
      if (layouts.find(arg_idx) == layouts.end()) continue;
      const std::string spec = layouts.lookup(arg_idx).sharding_spec(i);
      if (spec != Layout::kUnshardedDim) {
        spec_set.insert(spec);
      }
    }
    if (spec_set.size() == 1) {
      merged_specs[i] = *spec_set.begin();
    } else {
      merged_specs[i] = Layout::kUnshardedDim;
    }
  }

  // Deduplicate same usage of mesh dims. [x,x] -> [unsharded, unsharded]
  absl::flat_hash_map<std::string, int> counter;
  for (const std::string& spec : merged_specs) counter[spec] += 1;
  for (std::string& spec : merged_specs) {
    if (counter[spec] > 1) {
      spec = Layout::kUnshardedDim;
    }
  }
  return Layout::GetLayout(merged_specs, mesh);
}

// Choose an intermediate layout to relayout. Picks the most frequently
// sharded mesh dimension for every batch dimension, then deduplicates (n-1)
// of all repeated mesh dimensions, leaving the rightmost duplicate sharded
//
// Note that this assumes the number of batch dims for every batchable
// tensor is the same and is enforced before this
//
// Examples:
// Given layouts: [x,y],[x,z],[y,z], produces [x,z]
// Deduplication: [x,x] -> will become [*, x]
StatusOr<Layout> IntermediateBatchLayout(
    const std::vector<Layout>& operand_layouts,
    const llvm::DenseMap<int, int>& batchable_operands,
    const std::vector<Layout>& output_layouts,
    const llvm::DenseMap<int, int>& batchable_outputs, const Mesh& mesh) {
  if (batchable_operands.empty()) {
    return errors::Unimplemented(
        llvm::formatv("There must be at least one batchable operand").str());
  }
  int first_batcharg_index = batchable_outputs.begin()->first;
  int batch_rank = operand_layouts[first_batcharg_index].rank() -
                   batchable_operands.find(first_batcharg_index)->second;

  std::vector<std::string> batch_specs(batch_rank, Layout::kUnshardedDim);

  // For each batch dimension, finds the most commonly used mesh dimension
  // and sets that to batch_specs[i].
  for (int i = 0; i < batch_rank; ++i) {
    std::string mesh_dim = Layout::kUnshardedDim;
    int max_count = 0;
    absl::flat_hash_map<std::string, int> counter;
    // add operand counts
    for (auto const& idx_and_unused : batchable_operands) {
      auto const& idx = idx_and_unused.first;
      std::string spec = operand_layouts[idx].sharding_spec(i);
      if (spec != Layout::kUnshardedDim) counter[spec]++;
      if (counter[spec] > max_count) {
        max_count = counter[spec];
        mesh_dim = spec;
      }
    }
    // add output counts
    for (auto const& idx_and_unused : batchable_outputs) {
      auto const& idx = idx_and_unused.first;
      std::string spec = output_layouts[idx].sharding_spec(i);
      if (spec != Layout::kUnshardedDim) counter[spec]++;
      if (counter[spec] > max_count) {
        max_count = counter[spec];
        mesh_dim = spec;
      }
    }
    batch_specs[i] = mesh_dim;
  }
  // deduplicate
  absl::flat_hash_map<std::string, int> counter;
  for (const std::string& spec : batch_specs) counter[spec] += 1;
  for (std::string& spec : batch_specs) {
    if (counter[spec] > 1) {
      counter[spec]--;
      spec = Layout::kUnshardedDim;
    }
  }
  return Layout::GetLayout(batch_specs, mesh);
}
}  // namespace

// Relayout all operands that have batch dimensions to batch sharded
// The outputs will get the correct inferred shape from the operands
StatusOr<mlir::Operation*> DataparallelSPMDExpander::RelayoutOperandsAndOutputs(
    mlir::Operation* op, const std::vector<Layout>& operand_layouts,
    const std::vector<Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(
      const Layout intermediate_batch_layout,
      IntermediateBatchLayout(operand_layouts, batchable_operands_,
                              output_layouts, batchable_outputs_, mesh));
  // Relayout batchable operands
  for (auto i = 0; i < operand_layouts.size(); ++i) {
    // Relayout operands that have a batch dimension to intermediate layout
    if (batchable_operands_.find(i) != batchable_operands_.end()) {
      int replicated_rank =
          ValueRank(op->getOperand(i)) - intermediate_batch_layout.rank();
      TF_ASSIGN_OR_RETURN(
          auto new_layout,
          ConcatenateLayouts(intermediate_batch_layout,
                             Layout::ReplicatedOnMesh(mesh, replicated_rank)));
      TF_ASSIGN_OR_RETURN(
          const auto new_operand,
          EmitRelayout(op->getOperand(i), operand_layouts[i], new_layout));
      op->setOperand(i, new_operand);
    }
  }
  // Expand to local shape
  op = InferSPMDExpandedLocalShape(op);

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  // Track the op that comes last after splitting.
  mlir::Operation* last_op_after_splitting = op;

  // Relayout batchable outputs
  for (auto i = 0; i < output_layouts.size(); ++i) {
    // Relayout to batch shard if tensor has batch dim
    if (batchable_outputs_.find(i) != batchable_outputs_.end()) {
      int replicated_rank =
          ValueRank(op->getResult(i)) - intermediate_batch_layout.rank();
      TF_ASSIGN_OR_RETURN(
          auto new_layout,
          ConcatenateLayouts(intermediate_batch_layout,
                             Layout::ReplicatedOnMesh(mesh, replicated_rank)));
      TF_ASSIGN_OR_RETURN(auto new_output,
                          EmitRelayout(op->getOpResult(i), new_layout,
                                       output_layouts[i], &newly_created_ops));
      generated_outputs.emplace_back(new_output);
      generated_types.emplace_back(new_output.getType());
      if (last_op_after_splitting->isBeforeInBlock(
              new_output.getDefiningOp())) {
        last_op_after_splitting = new_output.getDefiningOp();
      }
    } else {
      generated_outputs.push_back(op->getResult(i));
      generated_types.push_back(op->getResult(i).getType());
    }
  }
  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(last_op_after_splitting);

  // Tie all outputs together with identity_n
  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);
  newly_created_ops.insert(identity_op);
  for (int i = 0; i < output_layouts.size(); ++i) {
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);
  }

  return identity_op.getOperation();
}

StatusOr<mlir::Operation*> DataparallelSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  // Check all input and output are batch parallel
  if (!AllBatchParallel(operand_layouts, batchable_operands_) ||
      !AllBatchParallel(output_layouts, batchable_outputs_)) {
    return errors::Unimplemented(
        llvm::formatv("All operands and outputs must be batch parallel.")
            .str());
  }
  // Check that the rank of batch dimensions are same for all batchable tensors
  if (!SameBatchRank(operand_layouts, batchable_operands_) ||
      !SameBatchRank(output_layouts, batchable_outputs_)) {
    return errors::Unimplemented(
        llvm::formatv("All operands and outputs with batch dimensions must "
                      "have same batch dimension rank")
            .str());
  }
  if (AllReplicated(output_layouts) && AllReplicated(operand_layouts))
    return InferSPMDExpandedLocalShape(op);
  return RelayoutOperandsAndOutputs(op, operand_layouts, output_layouts);
}

// Take all layouts of batchable operands, and merge them to produce a single
// layout for all batchable outputs.
StatusOr<llvm::DenseMap<int, Layout>>
DataparallelSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> output_layouts;
  // Compute output layouts
  if (AnyLayoutExist(input_layouts, batchable_operands_)) {
    TF_ASSIGN_OR_RETURN(
        const Layout& batch_output_layout,
        MergeBatchLayouts(input_layouts, batchable_operands_, mesh));
    for (const auto& output_and_index : llvm::enumerate(op->getOpResults())) {
      const int output_index = output_and_index.index();
      auto output = output_and_index.value();
      int rank = ValueRank(output);
      if (batchable_outputs_.find(output_index) != batchable_outputs_.end()) {
        int replicated_rank = batchable_outputs_[output_index];
        TF_ASSIGN_OR_RETURN(auto new_layout,
                            ConcatenateLayouts(batch_output_layout,
                                               Layout::ReplicatedOnMesh(
                                                   mesh, replicated_rank)));
        output_layouts[output_index] = new_layout;
      } else {
        output_layouts[output_index] = Layout::ReplicatedOnMesh(mesh, rank);
      }
    }
  }
  return output_layouts;
}

// Take all layouts of batchable outputs, and merge them to produce a single
// layout for all batchable operands.
StatusOr<llvm::DenseMap<int, Layout>>
DataparallelSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts;
  // Compute input layouts in the following way: For operand indices that
  // have a batch dimension, batch shard it the same way as the output layouts.
  // Otherwise, replicate.
  if (AnyLayoutExist(output_layouts, batchable_outputs_)) {
    TF_ASSIGN_OR_RETURN(
        const Layout& batch_operand_layout,
        MergeBatchLayouts(output_layouts, batchable_outputs_, mesh));

    for (const auto& operand_and_index : llvm::enumerate(op->getOperands())) {
      const int operand_index = operand_and_index.index();
      auto operand = operand_and_index.value();
      int rank = ValueRank(operand);
      if (batchable_operands_.find(operand_index) !=
          batchable_operands_.end()) {
        int replicated_rank = batchable_operands_[operand_index];
        TF_ASSIGN_OR_RETURN(auto new_layout,
                            ConcatenateLayouts(batch_operand_layout,
                                               Layout::ReplicatedOnMesh(
                                                   mesh, replicated_rank)));
        input_layouts[operand_index] = new_layout;
      } else {
        input_layouts[operand_index] = Layout::ReplicatedOnMesh(mesh, rank);
      }
    }
  }
  return input_layouts;
}
}  // namespace dtensor
}  // namespace tensorflow
