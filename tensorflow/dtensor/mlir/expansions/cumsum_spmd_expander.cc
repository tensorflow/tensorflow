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

#include "tensorflow/dtensor/mlir/expansions/cumsum_spmd_expander.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Extract `axis` tensor from Cumsum op and return it's positive value, since
// it can be a negative index.
StatusOr<int64_t> GetAxisDimension(mlir::Operation* op) {
  auto cumsum = llvm::dyn_cast<mlir::TF::CumsumOp>(op);
  if (cumsum == nullptr) {
    return errors::Internal(
        absl::StrCat("Expected Cumsum op but got : ", OpName(op)).c_str());
  }
  TF_ASSIGN_OR_RETURN(int64_t axis_dim,
                      ExtractConstIntFromValue(cumsum.axis()));
  int64_t tensor_rank = ValueRank(cumsum.x());
  // Axis can be in range [-tensor_rank, tensor_rank), so we add tensor_rank
  // to wrap it around.
  if (axis_dim >= -tensor_rank && axis_dim < 0) {
    axis_dim += tensor_rank;
  } else if (axis_dim < -tensor_rank || axis_dim >= tensor_rank) {
    return errors::InvalidArgument(
        "Invalid axis; expected a value in [-tensor_rank, tensor_rank)");
  }
  return axis_dim;
}

}  // namespace

StatusOr<mlir::Operation*> CumsumSPMDExpander::ExpandOp(mlir::Operation* op) {
  StatusOr<int64_t> axis_dim = GetAxisDimension(op);
  if (!axis_dim.ok()) return axis_dim.status();

  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  assert(output_layout);

  // Our intermediate computation layout is the output layout with
  // the axis dimension replicated. So set both the operand and output layout
  // to this intermediate layout.
  Layout intermediate_layout = output_layout->GetLayoutWithReducedDims(
      {axis_dim.value()}, /*keep_dims=*/true);

  // Relayout operand to intermediate layout.
  mlir::OpBuilder builder(op);
  const auto operand = op->getOperand(0);
  TF_ASSIGN_OR_RETURN(auto operand_layout, ExtractLayoutFromOperand(operand));
  if (!operand_layout)
    return errors::InvalidArgument(
        "input layout of Cumsum op must be known before SPMD "
        "expansion.");

  TF_ASSIGN_OR_RETURN(
      const auto new_operand,
      EmitRelayout(operand, operand_layout.value(), intermediate_layout));
  op->setOperand(0, new_operand);

  op = InferSPMDExpandedLocalShape(op);

  // Relayout output to intermediate layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(op);
  TF_ASSIGN_OR_RETURN(auto final_output,
                      EmitRelayout(op->getOpResult(0), intermediate_layout,
                                   output_layout.value(), &newly_created_ops));
  op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
  return final_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> CumsumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(int64_t axis_dim, GetAxisDimension(op));

  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto input_layout = input_layouts.lookup(0);
  return llvm::DenseMap<int, Layout>(
      {{0, input_layout.GetLayoutWithReducedDims({axis_dim},
                                                 /*keep_dims=*/true)}});
}

StatusOr<llvm::DenseMap<int, Layout>> CumsumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(int64_t axis_dim, GetAxisDimension(op));

  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();
  auto output_layout = output_layouts.lookup(0);
  return llvm::DenseMap<int, Layout>(
      {{0, output_layout.GetLayoutWithReducedDims({axis_dim},
                                                  /*keep_dims=*/true)}});
}

}  // namespace dtensor
}  // namespace tensorflow
