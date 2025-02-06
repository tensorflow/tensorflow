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

#include "tensorflow/dtensor/mlir/expansions/trivial_spmd_expander.h"

#include <cassert>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> TerminatorSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto terminator_op = llvm::cast<mlir::tf_device::ReturnOp>(op);
  auto parent_op = op->getParentOp();
  auto output_types = llvm::to_vector<8>(terminator_op.getOperandTypes());
  assert(output_types.size() == parent_op->getNumResults());

  for (const auto& output_type_and_index : llvm::enumerate(output_types)) {
    const int index = output_type_and_index.index();
    const auto& type = output_type_and_index.value();
    parent_op->getResult(index).setType(type);
  }
  return op;
}

StatusOr<mlir::Operation*> MetadataSPMDExpander::ExpandOp(mlir::Operation* op) {
  for (auto operand : op->getOperands()) {
    TF_ASSIGN_OR_RETURN(auto input_layout, ExtractLayoutFromOperand(operand));
    if (!input_layout.has_value())
      return errors::Internal(
          "All input layouts to Metadata op must be specified at SPMD "
          "expansion.");

    if (!input_layout->IsFullyReplicated())
      return errors::InvalidArgument(
          "Metadata ops like tf.BroadcastGradientArgs op must have replicated "
          "input layouts.");
  }

  TF_ASSIGN_OR_RETURN(auto result_layouts, ExtractLayoutFromOp(op));
  for (const auto& layout : result_layouts) {
    if (!layout.has_value())
      return errors::Internal(
          "All op result layouts of Metadata op must be specified for SPMD "
          "expansion.");

    if (!layout->IsFullyReplicated()) {
      return errors::InvalidArgument(
          "Metadata ops like tf.BroadcastGradientArgs op must have replicated "
          "output layouts.");
    }
  }
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>>
MetadataSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (const auto& result_and_index : llvm::enumerate(op->getOpResults())) {
    const int index = result_and_index.index();
    auto result = result_and_index.value();
    output_layouts.insert(
        {index, Layout::ReplicatedOnMesh(mesh, ValueRank(result))});
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
MetadataSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (const auto& operand_and_index : llvm::enumerate(op->getOperands())) {
    const int index = operand_and_index.index();
    auto operand = operand_and_index.value();
    input_layouts.insert(
        {index, Layout::ReplicatedOnMesh(mesh, ValueRank(operand))});
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
