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

#include "tensorflow/dtensor/mlir/expansions/control_flow_spmd_expander.h"

#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> WhileRegionSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  assert(op->getNumOperands() == op->getNumResults());
  // Set the type for the results of the WhileRegion explicitly.
  //
  // Normally we would use InferSPMDExpandedLocalShape for this, but that
  // function requires the op to either have a type inference interface
  // (which WhileRegion does not) or a TensorFlow ShapeFn (WhileRegion is not
  // a TensorFlow op). During the standard MLIR shape inference pass this op
  // is handled by a special case in InferShapeForSingleOperation.
  for (int i = 0; i < op->getNumOperands(); ++i)
    op->getResult(i).setType(op->getOperand(i).getType());

  auto while_op = llvm::cast<mlir::TF::WhileRegionOp>(op);
  for (const auto& data :
       llvm::enumerate(llvm::zip(while_op.getCond().front().getArguments(),
                                 while_op.getBody().front().getArguments()))) {
    const int index = data.index();
    mlir::BlockArgument cond_arg = std::get<0>(data.value());
    mlir::BlockArgument body_arg = std::get<1>(data.value());
    cond_arg.setType(while_op.getOperand(index).getType());
    body_arg.setType(while_op.getOperand(index).getType());
  }

  return op;
}

StatusOr<llvm::DenseMap<int, Layout>>
WhileRegionSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(
      "WhileRegion does not support compute layouts. This should not be "
      "called.");
}

StatusOr<llvm::DenseMap<int, Layout>>
WhileRegionSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(
      "WhileRegion does not support compute layouts. This should not be "
      "called.");
}

StatusOr<mlir::Operation*> IfRegionSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto if_op = llvm::cast<mlir::TF::IfRegionOp>(op);
  for (mlir::Value result : if_op->getResults()) {
    auto result_layout_op = llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
        *result.getUsers().begin());
    if (!result_layout_op)
      return errors::InvalidArgument(
          "Missing layout of If op result during SPMD expansion.");

    const Layout layout = result_layout_op.getLayout();
    if (!layout.IsFullyReplicated()) {
      const auto global_shape = result_layout_op.getGlobalShape();
      if (!global_shape)
        return errors::InvalidArgument(
            "Shape of If op must be statically known for SPMD expansion.");

      result.setType(mlir::RankedTensorType::get(
          layout.LocalShapeFromGlobalShape(*global_shape),
          mlir::cast<mlir::TensorType>(result.getType()).getElementType()));
    }
  }
  return op;
}

StatusOr<llvm::DenseMap<int, Layout>>
IfRegionSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // No-op for forward propagation.
  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
IfRegionSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // Layout propagation for TF::IfRegion op is no-op. Actual layout
  // propagation logic depends on layout propgation of ops inside the
  // then/else regions of the IfRegion op.
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)))}});
}

}  // namespace dtensor
}  // namespace tensorflow
