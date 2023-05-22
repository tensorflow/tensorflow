/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "tf-tpu-annotate-dynamic-shape-inputs"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_TPUANNOTATEDYNAMICSHAPEINPUTSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class TPUAnnotateDynamicShapeInputsPass
    : public impl::TPUAnnotateDynamicShapeInputsPassBase<
          TPUAnnotateDynamicShapeInputsPass> {
  void runOnOperation() override;
};

// Finds op that created a given value. If the value is a BlockArgument, this
// returns the owner of the Block.
Operation* GetOpOfValue(Value value) {
  if (auto block_arg = value.dyn_cast<BlockArgument>())
    return block_arg.getOwner()->getParentOp();

  return value.getDefiningOp();
}

void TPUAnnotateDynamicShapeInputsPass::runOnOperation() {
  getOperation().walk([&](tf_device::ClusterFuncOp cluster_func_op) {
    Builder builder(cluster_func_op->getContext());
    // Skip non-tpu device cluster_func.
    auto cluster_id =
        cluster_func_op->getAttrOfType<StringAttr>(TF::kReplicationInfoAttr);
    if (!cluster_id) return WalkResult::advance();

    llvm::SmallVector<int, 4> dynamic_shape_arg_index;

    // Traverse the operands of the cluster func op and find which operand
    // is returned by TPUAnnotateTensorsWithDynamicShapeOp.
    for (const auto& cluster_func_operand :
         llvm::enumerate(cluster_func_op.getOperands())) {
      auto device_launch_op = llvm::dyn_cast<tf_device::LaunchOp>(
          GetOpOfValue(cluster_func_operand.value()));
      if (!device_launch_op) continue;
      for (auto result : llvm::zip(
               device_launch_op.getResults(),
               device_launch_op.GetBody().getTerminator()->getOperands())) {
        if (std::get<0>(result) == cluster_func_operand.value() &&
            llvm::isa<TF::TPUAnnotateTensorsWithDynamicShapeOp>(
                std::get<1>(result).getDefiningOp())) {
          dynamic_shape_arg_index.push_back(cluster_func_operand.index());
        }
      }
    }

    cluster_func_op->setAttr(TF::kDynamicArgIndexAttr,
                             builder.getI32ArrayAttr(dynamic_shape_arg_index));

    FlatSymbolRefAttr func_attr = cluster_func_op.getFuncAttr();
    func::FuncOp func =
        cluster_func_op->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
            func_attr.getValue());

    // Update the marked argument with dynamic shapes.
    for (int index : dynamic_shape_arg_index) {
      BlockArgument arg = func.getArgument(index);
      auto inputType = arg.getType().dyn_cast<RankedTensorType>();
      // Only rank 1 tensor is supported for now.
      if (!inputType || inputType.getRank() != 1) continue;
      auto shape = llvm::to_vector<4>(inputType.getShape());
      llvm::SmallVector<int64_t, 4> bounds(shape.begin(), shape.end());
      // Mark the dim as dynamic dim.
      shape[0] = ShapedType::kDynamic;
      auto extensions =
          mhlo::TypeExtensionsAttr::get(func->getContext(), bounds);
      auto resultType =
          RankedTensorType::get(shape, inputType.getElementType(), extensions);
      arg.setType(resultType);
    }
    llvm::SmallVector<Type, 8> arg_types;
    for (auto arg : func.getArguments()) arg_types.push_back(arg.getType());
    func.setType(
        FunctionType::get(func.getContext(), arg_types,
                          func.front().getTerminator()->getOperandTypes()));
    return WalkResult::advance();
  });

  // Remove the annotated op after since it is just a placeholder.
  getOperation().walk([&](Operation* op) {
    if (llvm::isa<TF::TPUAnnotateTensorsWithDynamicShapeOp>(op)) {
      for (auto result : llvm::zip(op->getOperands(), op->getResults())) {
        std::get<1>(result).replaceAllUsesWith(std::get<0>(result));
      }
      op->erase();
    }
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTPUAnnotateDynamicShapeInputsPass() {
  return std::make_unique<TPUAnnotateDynamicShapeInputsPass>();
}
}  // namespace TFTPU
}  // namespace mlir
