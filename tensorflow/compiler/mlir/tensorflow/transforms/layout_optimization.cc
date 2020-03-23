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

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"

#define DEBUG_TYPE "tf-layout-optimization"

namespace mlir {
namespace TF {

namespace {

// LayoutAssignmentPass assigns optimal data layout (data format) for all
// layout sensitive operations.
class LayoutAssignmentPass : public FunctionPass<LayoutAssignmentPass> {
 public:
  LayoutAssignmentPass() = default;
  explicit LayoutAssignmentPass(const std::string& force_data_format) {
    force_data_format_ = force_data_format;
  }

  LayoutAssignmentPass(const LayoutAssignmentPass& pass) {}

  void runOnFunction() final;

 private:
  // Force a specified data format for all layout sensitive operations.
  Option<std::string> force_data_format_{
      *this, "force-data-format",
      llvm::cl::desc("Force data format for all layout sensitive ops")};
};

// MoveTransposesPass moves all Transpose ops to the beginning or to the end of
// the basic block where they are defined. This will allow canonicalzer to
// delete redundant transposes.
class MoveTransposesPass : public FunctionPass<MoveTransposesPass> {
 public:
  enum class Direction { kBegin, kEnd };

  MoveTransposesPass() = default;
  explicit MoveTransposesPass(Direction direction) { direction_ = direction; }
  MoveTransposesPass(const MoveTransposesPass& pass) {}

  void runOnFunction() final;

 private:
  Option<Direction> direction_{
      *this, "direction",
      llvm::cl::desc("Move transposes to the beginning or the end of the block "
                     "where they are defined."),
      llvm::cl::values(
          clEnumValN(Direction::kBegin, "begin", "beginning of the block"),
          clEnumValN(Direction::kEnd, "end", "end of the block"))};
};

using Permutation = SmallVector<int32_t, 4>;

Permutation GetDataFormatPermutation(StringRef from_data_format,
                                     StringRef to_data_format) {
  if (from_data_format == "NHWC" && to_data_format == "NCHW") {
    return {0, 3, 1, 2};
  } else if (from_data_format == "NCHW" && to_data_format == "NHWC") {
    return {0, 2, 3, 1};
  } else {
    llvm_unreachable("Unknown data format combination");
  }
}

void LayoutAssignmentPass::runOnFunction() {
  FuncOp func = getFunction();

  // Get runtime devices information from the closest parent module.
  RuntimeDevices devices;
  if (failed(::tensorflow::GetDevicesFromOp(func.getParentOfType<ModuleOp>(),
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
    OpBuilder builder(op->getBlock());

    auto perm_attr = [&](Permutation permutation) -> DenseIntElementsAttr {
      auto perm_ty = RankedTensorType::get({4}, builder.getIntegerType(32));
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
    TransposeOp transpose;
    if (!transpose_ops.empty()) {
      transpose = transpose_ops.pop_back_val();
      transpose.getOperation()->moveBefore(op);
      transpose.setOperand(0, operand.get());
      transpose.setOperand(1, permutation_op);
    } else {
      transpose =
          builder.create<TransposeOp>(loc, operand.get(), permutation_op);
    }

    operand.set(transpose);
  }

  // Remove unused transpose operations.
  while (!transpose_ops.empty()) {
    TransposeOp transpose = transpose_ops.pop_back_val();
    transpose.erase();
  }
}

// Move Transpose operations that permute `op` operands after the `op`.
void MoveTransposeAfter(Operation* op, SmallVector<Operation*, 8>* work_list) {
  // Indices of operands and results that depend on data layout.
  SmallVector<unsigned, 4> layout_dependent_operands;
  SmallVector<unsigned, 4> layout_dependent_results;

  auto fold_operands = dyn_cast<FoldOperandsTransposeInterface>(op);
  bool layout_agnostic = op->hasTrait<OpTrait::TF::LayoutAgnostic>();

  if (fold_operands) {
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

  // Check if we can fold transpose into the operation.
  if (fold_operands) {
    SmallVector<int64_t, 8> permutation;

    auto attr = permutation_op.value().cast<DenseElementsAttr>();
    for (auto value : attr.getIntValues())
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
  builder.setInsertionPoint(op);

  for (unsigned idx : layout_dependent_results) {
    OpResult result = op->getResult(idx);

    // Forward operand type only for layout agnostic operations, operations with
    // custom folding will update the result type in `FoldOperandsPermutation`.
    if (layout_agnostic) result.setType(op->getOperand(0).getType());

    // Try to push transpose further down.
    for (Operation* user : result.getUsers()) work_list->push_back(user);

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

void MoveTransposesPass::runOnFunction() {
  FuncOp func = getFunction();

  SmallVector<Operation*, 8> work_list;

  func.walk([&](TransposeOp transpose) {
    if (direction_ == Direction::kBegin) {
      // Try to push transpose before the operand operation.
      for (auto operand : transpose.getOperands()) {
        if (auto op = operand.getDefiningOp()) work_list.push_back(op);
      }
    } else {
      // Try to push transpose after the user operation.
      for (Operation* user : transpose.y().getUsers()) {
        work_list.push_back(user);
      }
    }
  });

  while (!work_list.empty()) {
    Operation* op = work_list.pop_back_val();
    if (direction_ == Direction::kBegin) {
      MoveTransposeBefore(op, &work_list);
    } else if (direction_ == Direction::kEnd) {
      MoveTransposeAfter(op, &work_list);
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
  using Direction = MoveTransposesPass::Direction;

  // Assign optimal layout for layout sensitive ops.
  pm.addPass(std::make_unique<LayoutAssignmentPass>(options.force_data_format));

  // Move transposes to the beginning of the block and try to fold them.
  pm.addPass(std::make_unique<MoveTransposesPass>(Direction::kBegin));

  // Move transposes to the end of the block and try to fold them.
  pm.addPass(std::make_unique<MoveTransposesPass>(Direction::kEnd));
}

static PassRegistration<LayoutAssignmentPass> layout_assignment(
    "tf-layout-assignment", "Layout assignment pass");
static PassRegistration<MoveTransposesPass> move_transposes(
    "tf-move-transposes", "Move transposes pass");

static mlir::PassPipelineRegistration<LayoutOptimizationPipelineOptions>
    pipeline("tf-layout-optimization",
             "Assigns optimal data layout to all layout sensitive operations "
             "and cancel redundant transpose operations.",
             CreateLayoutOptimizationPipeline);

}  // namespace TF
}  // namespace mlir
