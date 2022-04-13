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

#include "tensorflow/dtensor/mlir/expansions/tensorlist_getitem_spmd_expander.h"

#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> TensorListGetItemSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  TF_ASSIGN_OR_RETURN(const auto output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  // Do a final relayout to the correct output layout in case there are any
  // differences between layout of `handle` and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(mlir::Value output,
                      EmitRelayout(op->getOpResult(0), operand_layouts[0],
                                   output_layout, &newly_created_ops));

  op->getOpResult(0).replaceAllUsesExcept(output, newly_created_ops);
  return InferSPMDExpandedLocalShape(output.getDefiningOp());
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorListGetItemSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // Prefer the output to be the layout of the list handle.
  if (input_layouts.find(0) != input_layouts.end()) {
    return llvm::DenseMap<int, Layout>({{0, input_layouts.lookup(0)}});
  }
  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorListGetItemSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // Do not infer any layout to the operands.
  return llvm::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace tensorflow
