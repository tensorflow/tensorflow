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

#include "tensorflow/dtensor/mlir/expansions/split_spmd_expander.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Merges the output layouts to a common layout for Split/SplitV.
//
// Split has multiple outputs, so we first calculate a common output layout that
// we can use for passing backwards and then merge the remaining layouts.
StatusOr<Layout> MergeLayoutsForSplitOutput(
    int64_t split_dim, const llvm::DenseMap<int, Layout>& layouts) {
  assert(!layouts.empty());
  const Layout& first_layout = layouts.begin()->getSecond();
  std::vector<std::string> sharding_specs = first_layout.sharding_spec_strs();

  // Merge remaining layouts. If there is a conflicting sharding, then set the
  // dim to replicated.
  for (auto it = layouts.begin(); it != layouts.end(); ++it) {
    const Layout& output_layout = it->getSecond();
    for (int dim = 0; dim < output_layout.rank(); ++dim) {
      if (Layout::IsShardedDimension(output_layout.sharding_spec(dim)) &&
          Layout::IsShardedDimension(sharding_specs[dim]) &&
          output_layout.sharding_spec(dim) != sharding_specs[dim]) {
        sharding_specs[dim] = Layout::kUnshardedDim;
      }
    }
  }
  // Force the split_dim to be unsharded.
  sharding_specs[split_dim] = Layout::kUnshardedDim;
  return Layout::GetLayout(sharding_specs, first_layout.mesh());
}

// Retrieves the value of the split_dim operand adjusted based on the input
// rank. The split_dim operand's value can be [-rank(input), rank(input)), which
// is adjusted to a positive value.
StatusOr<int64_t> GetAdjustedSplitDim(mlir::Value split_dim_value,
                                      mlir::Value input_value) {
  TF_ASSIGN_OR_RETURN(int64_t split_dim,
                      ExtractConstIntFromValue(split_dim_value));
  if (split_dim < 0) {
    int rank = ValueRank(input_value);
    if (rank == -1) {
      return errors::InvalidArgument("Input operand has rank -1.");
    }
    split_dim += rank;
  }
  return split_dim;
}

}  // namespace

StatusOr<mlir::Operation*> SplitSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto split_op = mlir::cast<mlir::TF::SplitOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(split_op.getValue()));
  TF_ASSIGN_OR_RETURN(
      const int64_t split_dim,
      GetAdjustedSplitDim(split_op.getSplitDim(), split_op.getValue()));

  if (Layout::IsShardedDimension(input_layout.sharding_spec(split_dim))) {
    return errors::InvalidArgument(
        "Spliting over sharded dimension is not supported.");
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> SplitSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto split_op = mlir::cast<mlir::TF::SplitOp>(op);
  llvm::DenseMap<int, Layout> output_layouts(split_op.getNumResults());
  if (input_layouts.find(1) != input_layouts.end()) {
    const Layout& suggested_layout = input_layouts.lookup(1);
    for (int i = 0; i < split_op.getNumResults(); ++i) {
      output_layouts[i] = suggested_layout;
    }
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>> SplitSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto split_op = mlir::cast<mlir::TF::SplitOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(split_op.getNumOperands());
  // axis
  input_layouts[0] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);

  if (!output_layouts.empty()) {
    // Split has multiple outputs, first calculate a common output layout that
    // we can use for passing backwards.
    TF_ASSIGN_OR_RETURN(
        const int64_t split_dim,
        GetAdjustedSplitDim(split_op.getSplitDim(), split_op.getValue()));
    TF_ASSIGN_OR_RETURN(const Layout common_output_layout,
                        MergeLayoutsForSplitOutput(split_dim, output_layouts));
    // value
    input_layouts[1] = common_output_layout;
  }

  return input_layouts;
}

StatusOr<mlir::Operation*> SplitVSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto split_v_op = mlir::cast<mlir::TF::SplitVOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(split_v_op.getValue()));
  TF_ASSIGN_OR_RETURN(
      const int64_t split_dim,
      GetAdjustedSplitDim(split_v_op.getSplitDim(), split_v_op.getValue()));

  if (Layout::IsShardedDimension(input_layout.sharding_spec(split_dim))) {
    return errors::InvalidArgument(
        "Spliting over sharded dimension is not supported.");
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> SplitVSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto split_v_op = mlir::cast<mlir::TF::SplitVOp>(op);
  llvm::DenseMap<int, Layout> output_layouts(split_v_op.getNumResults());
  if (input_layouts.find(0) != input_layouts.end()) {
    const Layout& suggested_layout = input_layouts.lookup(0);
    for (int i = 0; i < split_v_op.getNumResults(); ++i) {
      output_layouts[i] = suggested_layout;
    }
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>> SplitVSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto split_v_op = mlir::cast<mlir::TF::SplitVOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(split_v_op.getNumOperands());
  // size_splits
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  // axis
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);

  if (!output_layouts.empty()) {
    // Split has multiple outputs, first calculate a common output layout that
    // we can use for passing backwards.
    TF_ASSIGN_OR_RETURN(
        const int64_t split_dim,
        GetAdjustedSplitDim(split_v_op.getSplitDim(), split_v_op.getValue()));
    TF_ASSIGN_OR_RETURN(const Layout common_output_layout,
                        MergeLayoutsForSplitOutput(split_dim, output_layouts));
    // value
    input_layouts[0] = common_output_layout;
  }

  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
