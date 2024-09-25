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

#include "tensorflow/dtensor/mlir/expansions/expanddims_spmd_expander.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> ExpandDimsExpander::ExpandOp(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> operand_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));
  mlir::TF::ExpandDimsOp expand_dims_op =
      mlir::cast<mlir::TF::ExpandDimsOp>(op);

  InferSPMDExpandedLocalShape(op);

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> global_output_shape,
      GetGlobalShapeOfValueFromDTensorLayout(expand_dims_op.getOutput()));

  // Compute current output layout (just input layout with unsharded on the
  // new dim);
  TF_ASSIGN_OR_RETURN(int64_t dim,
                      ExtractConstIntFromValue(expand_dims_op.getDim()));

  if (dim < 0) dim += global_output_shape.size();
  std::vector<std::string> sharding_specs(global_output_shape.size());
  for (int i = 0; i < global_output_shape.size(); ++i) {
    if (i < dim)
      sharding_specs[i] = operand_layout->sharding_spec(i);
    else if (i == dim)
      sharding_specs[i] = Layout::kUnshardedDim;
    else
      sharding_specs[i] = operand_layout->sharding_spec(i - 1);
  }
  TF_ASSIGN_OR_RETURN(const Layout current_output_layout,
                      Layout::GetLayout(sharding_specs, output_layout->mesh()));

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(
      mlir::Value output_value,
      EmitRelayout(expand_dims_op.getOutput(), current_output_layout,
                   *output_layout, &newly_created_ops));

  expand_dims_op.getOutput().replaceAllUsesExcept(output_value,
                                                  newly_created_ops);

  return output_value.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> ExpandDimsExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  auto expand_dims_op = mlir::cast<mlir::TF::ExpandDimsOp>(op);

  TF_ASSIGN_OR_RETURN(int64_t dim,
                      ExtractConstIntFromValue(expand_dims_op.getDim()));

  // Do not infer any output layout if no operand layout is present.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto input_layout = input_layouts.lookup(0);

  if (dim < 0) dim += input_layout.rank() + 1;
  std::vector<std::string> layout_sharding;

  for (int i = 0; i <= input_layout.rank(); ++i) {
    if (i == dim) layout_sharding.push_back(Layout::kUnshardedDim);
    if (i < input_layout.rank())
      layout_sharding.push_back(input_layout.sharding_spec(i));
  }
  TF_ASSIGN_OR_RETURN(auto inferred_output_layout,
                      Layout::GetLayout(layout_sharding, mesh));
  return llvm::DenseMap<int, Layout>({{0, inferred_output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> ExpandDimsExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();
  auto output_layout = output_layouts.lookup(0);

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  auto expand_dims_op = mlir::cast<mlir::TF::ExpandDimsOp>(op);

  TF_ASSIGN_OR_RETURN(int64_t dim,
                      ExtractConstIntFromValue(expand_dims_op.getDim()));

  if (dim < 0) dim += output_layout.rank();

  std::vector<std::string> layout_sharding;

  for (int i = 0; i < output_layout.rank(); ++i) {
    if (i == dim) continue;
    layout_sharding.push_back(output_layout.sharding_spec(i));
  }

  TF_ASSIGN_OR_RETURN(auto inferred_input_layout,
                      Layout::GetLayout(layout_sharding, mesh));
  auto input_axis_rank = ValueRank(expand_dims_op->getOperand(1));

  return llvm::DenseMap<int, Layout>(
      {{0, inferred_input_layout},
       {1, Layout::ReplicatedOnMesh(mesh, /*rank=*/input_axis_rank)}});
}

}  // namespace dtensor
}  // namespace tensorflow
