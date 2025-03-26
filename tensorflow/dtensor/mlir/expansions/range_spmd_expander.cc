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

#include "tensorflow/dtensor/mlir/expansions/range_spmd_expander.h"

#include "absl/status/status.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> RangeSPMDExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto layout, ExtractSingleLayoutFromOp(op));

  if (!layout)
    return absl::InvalidArgumentError(
        "layout of RangeOp must be known before SPMD expansion.");

  if (!layout->IsFullyReplicated())
    return absl::InternalError("Shared RangeOp is not supported yet.");

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> RangeSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always return a Replicated layout. This will always respect the consumer
  // requested layouts.
  return llvm::DenseMap<int, Layout>({{0, Layout::ReplicatedOnMesh(mesh, 1)}});
}

StatusOr<llvm::DenseMap<int, Layout>> RangeSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always assign a replicated layout to the operands.
  llvm::DenseMap<int, Layout> input_layouts;
  for (int i = 0; i < op->getNumOperands(); ++i)
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, /*rank=*/ValueRank(op->getOperand(i)));
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
