/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_layout_helper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

#define DEBUG_TYPE "tf-layout-optimization"

namespace mlir {
namespace TF {

namespace {

// Helper method that returns an op from 'transpose_ops' that match criteria
// for an 'operand' and 'permutation'
TransposeOp ReuseExistingTranspose(const OpOperand* operand,
                                   const SmallVector<int64_t, 4>& permutation,
                                   Operation* op, ConstOp permutation_op,
                                   SmallVector<TransposeOp, 2>* transpose_ops) {
  for (auto it = transpose_ops->begin(); it != transpose_ops->end(); ++it) {
    auto tranpose_op = *it;
    for (auto tranpose_operand : tranpose_op.getOperands()) {
      auto ranked_tranpose_type =
          tranpose_operand.getType().dyn_cast_or_null<RankedTensorType>();
      if (!ranked_tranpose_type) continue;
      if (ranked_tranpose_type.getRank() == permutation.size() &&
          operand->get().getType() ==
              ShuffleRankedTensorType(ranked_tranpose_type, permutation)) {
        TransposeOp transpose = tranpose_op;
        transpose.getOperation()->moveBefore(op);
        transpose.setOperand(0, operand->get());
        transpose.setOperand(1, permutation_op);
        transpose_ops->erase(it);
        return transpose;
      }
    }
  }
  return nullptr;
}

// LayoutAssignmentPass assigns optimal data layout (data format) for all
// layout sensitive operations.
class LayoutAssignmentPass
    : public LayoutAssignmentPassBase<LayoutAssignmentPass> {
 public:
  LayoutAssignmentPass() = default;
  explicit LayoutAssignmentPass(const std::string& force_data_format) {
    force_data_format_ = force_data_format;
  }

  LayoutAssignmentPass(const LayoutAssignmentPass& pass) {}

  void runOnOperation() final;
};

// MoveTransposesPass moves all Transpose ops to the beginning or to the end of
// the basic block where they are defined. This will allow canonicalzer to
// delete redundant transposes.
class MoveTransposesPass : public MoveTransposesPassBase<MoveTransposesPass> {
 public:
  MoveTransposesPass() = default;
  explicit MoveTransposesPass(MoveTransposeDirection direction,
                              bool fold_transpose_in_ops) {
    this->direction_ = direction;
    this->fold_transpose_in_ops_ = fold_transpose_in_ops;
  }
  MoveTransposesPass(const MoveTransposesPass& pass) {}

  void runOnOperation() final;
};

using Permutation = SmallVector<int64_t, 4>;

void LayoutAssignmentPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Get runtime devices information from the closest parent module.
  RuntimeDevices devices;
  if (failed(::tensorflow::GetDevicesFromOp(func->getParentOfType<ModuleOp>(),
                                            &devices)))
    return signalPassFailure();

  // If there is no runtime device information and data format is not explicitly
  // forced, there is nothing to do.
  if (devices.NumDevices() == 0 && force_data_format_.empty()) return;

  func.walk([&](LayoutSensitiveInterface layout_sensitive_interface) {
    // Get desired op data format.
    StringRef target_data_format = force_data_format_;
    if (target_data_format.empty()) {
      target_data_format = layout_sensitive_interface.GetOptimalLayout(devices);
    }

    // Skip ops that already use target data format.
    auto data_format = layout_sensitive_interface.data_format();
    if (data_format == target_data_format) return;

    // Transpose arguments into the target data format.
    Permutation args_permutation =
        GetDataFormatPermutation(data_format, target_data_format);

    // Transpose results back to the original data format.
    Permutation res_permutation =
        GetDataFormatPermutation(target_data_format, data_format);

    if (args_permutation.empty() || res_permutation.empty()) return;

    mlir::Operation* op = layout_sensitive_interface.getOperation();
    Location loc = op->getLoc();
    OpBuilder builder = OpBuilder::atBlockEnd(op->getBlock());

    auto perm_attr = [&](Permutation permutation) -> DenseIntElementsAttr {
      auto perm_ty = RankedTensorType::get({4}, builder.getIntegerType(64));
      return DenseIntElementsAttr::get(perm_ty, permutation);
    };

    // Change operation data format.
    if (failed(layout_sensitive_interface.UpdateDataFormat(target_data_format)))
      return;

    // Permute arguments into the target data format.
    builder.setInsertionPoint(op);
    auto arg_perm = builder.create<ConstOp>(loc, perm_attr(args_permutation));

    for (int64_t arg : layout_sensitive_interface.GetLayoutDependentArgs()) {
      op->setOperand(
          arg, builder.create<TransposeOp>(loc, op->getOperand(arg), arg_perm));
    }

    // Permute results back to the original data format.
    builder.setInsertionPointAfter(op);
    auto res_perm = builder.create<ConstOp>(loc, perm_attr(res_permutation));

    for (int64_t res : layout_sensitive_interface.GetLayoutDependentResults()) {
      OpResult result = op->getResult(res);

      auto transposed_res = builder.create<TransposeOp>(loc, result, res_perm);
      result.replaceAllUsesWith(transposed_res);
      transposed_res.setOperand(0, result);
    }
  });
}

// Move Transpose operations that permute `op` results before the `op`.
void MoveTransposeBefore(Operation* op, SmallVector<Operation*, 8>* work_list) {
  // TODO(ezhulenev): Move transpose across layout sensitive operations.
  if (!op->hasTrait<OpTrait::TF::LayoutAgnostic>()) return;

  // Transpose operations that use operation results.
  SmallVector<TransposeOp, 2> transpose_ops;

  // Constant operation that defines permutation indices for result transposes.
  ConstOp permutation_op;

  // All operation results must be used by transpose operations with the same
  // permutation indices.
  for (OpResult result : op->getResults()) {
    for (Operation* user : result.getUsers()) {
      // Result user must be a transpose operation.
      TransposeOp transpose = dyn_cast<TransposeOp>(user);
      if (!transpose) return;

      // With permutation defined by constant operation.
      ConstOp perm =
          dyn_cast_or_null<ConstOp>(transpose.getOperand(1).getDefiningOp());
      if (!perm) return;

      // With the same permutation indices.
      auto dense_elem_attr = perm.value().dyn_cast<DenseElementsAttr>();
      if (!dense_elem_attr) return;

      if (!permutation_op) permutation_op = perm;

      // Check that permutation matches for all result transposes.
      if (perm.value() != permutation_op.value()) return;

      // Add a transpose operation for later reuse.
      transpose_ops.push_back(transpose);
    }
  }

  // Nothing to do here.
  if (!permutation_op || transpose_ops.empty()) return;
  SmallVector<int64_t, 4> permutation;
  auto perm_attr = permutation_op.value().cast<DenseElementsAttr>();
  for (const auto& value : perm_attr.getValues<APInt>())
    permutation.push_back(value.getSExtValue());

  // We want to make sure the shape of the operand equals the transposed shape.
  // mismatch can happen if 'op' supports broadcasting and the operands have
  // different ranks.
  if (op->hasTrait<OpTrait::ResultsBroadcastableShape>()) {
    auto transpose_op = *transpose_ops.begin();
    auto result_type =
        transpose_op.getResult().getType().dyn_cast_or_null<ShapedType>();
    auto is_valid_move =
        llvm::all_of(op->getOperands(), [result_type](Value operand) -> bool {
          auto operand_type = operand.getType().dyn_cast_or_null<ShapedType>();
          return result_type && operand_type && result_type.hasRank() &&
                 operand_type.hasRank() &&
                 result_type.getRank() == operand_type.getRank();
        });
    if (!is_valid_move) return;
  }

  // At this point we checked that we can safely move Transpose node before
  // `op`, and bypass all result transposes.
  Location loc = op->getLoc();

  // Move constant op defining result permutation to the beginning of the block.
  permutation_op.getOperation()->moveBefore(&op->getBlock()->front());

  // Bypass Transpose nodes for all results.
  for (OpResult result : op->getResults()) {
    result.setType(cast<TransposeOp>(*result.getUsers().begin()).y().getType());
    for (Operation* transpose : result.getUsers()) {
      transpose->getResult(0).replaceAllUsesWith(result);
    }
  }

  // Maybe add a Transpose node for all operands (or reuse existing transposes).
  OpBuilder builder(op);
  builder.setInsertionPoint(op);

  for (OpOperand& operand : op->getOpOperands()) {
    // Try to push transpose further up.
    if (Operation* operand_op = operand.get().getDefiningOp())
      work_list->push_back(operand_op);

    // Try to reuse result transposes.
    TransposeOp transpose = ReuseExistingTranspose(
        &operand, permutation, op, permutation_op, &transpose_ops);
    // If no transpose available for using, create new one.
    if (!transpose)
      transpose =
          builder.create<TransposeOp>(loc, operand.get(), permutation_op);

    operand.set(transpose);
  }

  // Remove unused transpose operations.
  while (!transpose_ops.empty()) {
    TransposeOp transpose = transpose_ops.pop_back_val();
    transpose.erase();
  }
}

// Revert the permutation applied in `type`.
static mlir::ShapedType ReversePermuteShapedType(
    mlir::ShapedType type, ArrayRef<int64_t> permutation) {
  if (!type.hasRank()) return type;

  auto shape = type.getShape();
  SmallVector<int64_t, 4> new_shape(shape.size());

  for (int i = 0; i < permutation.size(); ++i) {
    int64_t index = permutation[i];
    assert(index < shape.size());
    new_shape[index] = shape[i];
  }

  return type.clone(new_shape);
}

// Move Transpose operations that permute `op` operands after the `op`.
void MoveTransposeAfter(Operation* op, SmallVector<Operation*, 8>* work_list,
                        bool fold_transpose_in_ops) {
  // Indices of operands and results that depend on data layout.
  SmallVector<unsigned, 4> layout_dependent_operands;
  SmallVector<unsigned, 4> layout_dependent_results;

  auto fold_operands = dyn_cast<FoldOperandsTransposeInterface>(op);
  bool layout_agnostic = op->hasTrait<OpTrait::TF::LayoutAgnostic>();

  if (fold_operands && fold_transpose_in_ops) {
    layout_dependent_operands = fold_operands.GetLayoutDependentArgs();
    layout_dependent_results = fold_operands.GetLayoutDependentResults();

  } else if (layout_agnostic) {
    // For layout agnostic operation (e.g. element wise operations) all operands
    // and results must have the same data layout.
    for (unsigned i = 0; i < op->getNumOperands(); ++i)
      layout_dependent_operands.push_back(i);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      layout_dependent_results.push_back(i);
  }

  // Transpose operations that are operands of the `op`.
  SmallVector<TransposeOp, 2> transpose_ops;

  // Constant operation that defines permutation indices for operand transposes.
  ConstOp permutation_op;

  // Layout dependent operands must be transpose operations with the same
  // permutation indices.
  for (unsigned idx : layout_dependent_operands) {
    OpOperand& operand = op->getOpOperand(idx);

    // Operand must be defined by a transpose op.
    TransposeOp transpose =
        dyn_cast_or_null<TransposeOp>(operand.get().getDefiningOp());
    if (!transpose) return;

    // With permutation defined by constant operation.
    ConstOp perm =
        dyn_cast_or_null<ConstOp>(transpose.getOperand(1).getDefiningOp());
    if (!perm) return;

    // With the same permutation indices.
    auto dense_elem_attr = perm.value().dyn_cast<DenseElementsAttr>();
    if (!dense_elem_attr) return;

    if (!permutation_op) permutation_op = perm;

    // Check that permutation matches for all result transposes.
    if (perm.value() != permutation_op.value()) return;

    // Add a transpose operation for later reuse only if it's used once.
    if (transpose.getResult().hasOneUse()) transpose_ops.push_back(transpose);
  }

  // Nothing to do here.
  if (!permutation_op) return;

  // All results after transpose must preserve the original result type.
  SmallVector<Type, 4> original_type(op->getNumResults());
  for (unsigned idx : layout_dependent_results)
    original_type[idx] = op->getResult(idx).getType();

  SmallVector<int64_t, 8> permutation;

  auto attr = permutation_op.value().cast<DenseElementsAttr>();
  for (const auto& value : attr.getValues<APInt>())
    permutation.push_back(value.getSExtValue());

  // Check if we can fold transpose into the operation.
  if (fold_operands && fold_transpose_in_ops) {
    SmallVector<int64_t, 8> permutation;

    auto attr = permutation_op.value().cast<DenseElementsAttr>();
    for (const auto& value : attr.getValues<APInt>())
      permutation.push_back(value.getSExtValue());

    if (failed(fold_operands.FoldOperandsPermutation(permutation))) return;
  }

  // At this point we checked that we can safely move Transpose node after
  // `op`, bypass all operands transposes, and transpose op results.
  Location loc = op->getLoc();

  // Move constant op defining result permutation to the beginning of the block.
  permutation_op.getOperation()->moveBefore(&op->getBlock()->front());

  // Bypass Transpose nodes for layout dependent operands.
  for (unsigned idx : layout_dependent_operands) {
    OpOperand& operand = op->getOpOperand(idx);
    TransposeOp transpose =
        dyn_cast<TransposeOp>(operand.get().getDefiningOp());
    operand.set(transpose.getOperand(0));
  }

  // Maybe add Transpose nodes for layout dependent results
  // (or reuse existing transposes).
  OpBuilder builder(op);
  builder.setInsertionPointAfter(op);

  for (unsigned idx : layout_dependent_results) {
    OpResult result = op->getResult(idx);

    // If the op is layout agnostic, the new result type can be generated by
    // reverting `permutation`. Otherwise, operations with custom folding will
    // update the result type in `FoldOperandsPermutation`.
    if (layout_agnostic)
      result.setType(ReversePermuteShapedType(
          result.getType().cast<ShapedType>(), permutation));

    // Try to push transpose further down.
    for (Operation* user : result.getUsers()) {
      if (!llvm::isa<TransposeOp>(user)) work_list->push_back(user);
    }

    // Try to reuse operand transposes.
    TransposeOp transpose;
    if (!transpose_ops.empty()) {
      transpose = transpose_ops.pop_back_val();
      transpose.getOperation()->moveBefore(op->getNextNode());
      transpose.setOperand(0, result);
      transpose.setOperand(1, permutation_op);
      transpose.getResult().setType(original_type[idx]);
    } else {
      transpose = builder.create<TransposeOp>(loc, result, permutation_op);
    }

    // Forward all users to the transpose operation.
    result.replaceAllUsesWith(transpose);
    transpose.setOperand(0, result);
  }

  // Remove unused transpose operations.
  while (!transpose_ops.empty()) {
    TransposeOp transpose = transpose_ops.pop_back_val();
    transpose.erase();
  }
}

void MoveTransposesPass::runOnOperation() {
  func::FuncOp func = getOperation();

  SmallVector<Operation*, 8> work_list;

  func.walk([&](TransposeOp transpose) {
    if (direction_ == MoveTransposeDirection::kBegin) {
      // Try to push transpose before the operand operation.
      for (auto operand : transpose.getOperands()) {
        if (auto op = operand.getDefiningOp()) work_list.push_back(op);
      }
    } else {
      // Try to push transpose after the user operation.
      for (Operation* user : transpose.y().getUsers()) {
        if (!llvm::isa<TransposeOp>(user)) work_list.push_back(user);
      }
    }
  });

  while (!work_list.empty()) {
    Operation* op = work_list.pop_back_val();
    if (direction_ == MoveTransposeDirection::kBegin) {
      MoveTransposeBefore(op, &work_list);
    } else if (direction_ == MoveTransposeDirection::kEnd) {
      MoveTransposeAfter(op, &work_list, fold_transpose_in_ops_);
    }
  }

  func.walk([&](TransposeOp transpose) {
    OpBuilder builder(transpose);
    SmallVector<Value, 1> fold_result;
    if (succeeded(builder.tryFold(transpose.getOperation(), fold_result))) {
      assert(fold_result.size() == 1);
      transpose.replaceAllUsesWith(fold_result[0]);
    }
  });
}

}  // namespace

void CreateLayoutOptimizationPipeline(
    OpPassManager& pm,  // NOLINT - MLIR contract is pass by mutable reference.
    const LayoutOptimizationPipelineOptions& options) {
  // Assign optimal layout for layout sensitive ops.
  pm.addPass(std::make_unique<LayoutAssignmentPass>(options.force_data_format));

  // Move transposes to the beginning of the block and try to fold them.
  pm.addPass(std::make_unique<MoveTransposesPass>(
      MoveTransposeDirection::kBegin, !options.skip_fold_transpose_in_ops));

  // Move transposes to the end of the block and try to fold them.
  pm.addPass(std::make_unique<MoveTransposesPass>(
      MoveTransposeDirection::kEnd, !options.skip_fold_transpose_in_ops));
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateLayoutAssignmentPass() {
  // This static is kind of hack, it hooks the pipeline registration for the
  // command line and piggy-back to the TableGen generated registration code.
  static mlir::PassPipelineRegistration<LayoutOptimizationPipelineOptions>
      pipeline("tf-layout-optimization",
               "Assigns optimal data layout to all layout sensitive operations "
               "and cancel redundant transpose operations.",
               CreateLayoutOptimizationPipeline);
  return std::make_unique<LayoutAssignmentPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateMoveTransposesPass() {
  return std::make_unique<MoveTransposesPass>();
}

}  // namespace TF
}  // namespace mlir
