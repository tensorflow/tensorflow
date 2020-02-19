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
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "tf-layout-optimization"

namespace mlir {
namespace TF {

namespace {

// LayoutAssignmentPass assigns optimal data layout (data format) for all
// layout sensitive operations.
class LayoutAssignmentPass : public FunctionPass<LayoutAssignmentPass> {
 public:
  LayoutAssignmentPass() = default;
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
  void runOnFunction() final;
};

using Permutation = SmallVector<int64_t, 4>;

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

Type PermuteRankedTensorType(Type type, Permutation permutation) {
  if (auto ranked_type = type.dyn_cast<RankedTensorType>()) {
    ArrayRef<int64_t> shape = ranked_type.getShape();
    assert(permutation.size() == shape.size());

    SmallVector<int64_t, 4> new_shape(permutation.size());
    for (size_t i = 0; i < permutation.size(); ++i) {
      new_shape[i] = shape[permutation[i]];
    }

    return RankedTensorType::get(new_shape, ranked_type.getElementType());
  }

  return type;
}

void LayoutAssignmentPass::runOnFunction() {
  FuncOp func = getFunction();

  // TODO(ezhulenev): LayoutSensitiveInterface should select the optimal data
  // layout if there is no explicitly forced data format.
  if (force_data_format_.empty()) return;

  func.walk([&](LayoutSensitiveInterface layout_sensitive_interface) {
    // Skip ops that already use target data format.
    auto data_format = layout_sensitive_interface.data_format();
    if (data_format == force_data_format_) return;

    // Transpose arguments into the target data format.
    Permutation args_permutation =
        GetDataFormatPermutation(data_format, force_data_format_);

    // Transpose results back to the original data format.
    Permutation res_permutation =
        GetDataFormatPermutation(force_data_format_, data_format);

    if (args_permutation.empty() || res_permutation.empty()) return;

    mlir::Operation* op = layout_sensitive_interface.getOperation();
    Location loc = op->getLoc();
    OpBuilder builder(op->getBlock());

    auto perm_attr = [&](Permutation permutation) -> DenseIntElementsAttr {
      auto perm_ty = RankedTensorType::get({4}, builder.getIntegerType(64));
      return DenseIntElementsAttr::get(perm_ty, permutation);
    };

    // Change operation data format.
    op->setAttr("data_format",
                StringAttr::get(force_data_format_, op->getContext()));

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
      result.setType(
          PermuteRankedTensorType(result.getType(), args_permutation));

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

void MoveTransposesPass::runOnFunction() {
  FuncOp func = getFunction();

  SmallVector<Operation*, 8> work_list;

  func.walk([&](TransposeOp transpose) {
    for (auto operand : transpose.getOperands()) {
      if (auto op = operand.getDefiningOp()) work_list.push_back(op);
    }
  });

  while (!work_list.empty()) {
    Operation* op = work_list.pop_back_val();
    MoveTransposeBefore(op, &work_list);
  }
}

}  // namespace

static PassRegistration<LayoutAssignmentPass> layout_assignment(
    "tf-layout-assignment", "Layout assignment pass");
static PassRegistration<MoveTransposesPass> move_transposes(
    "tf-move-transposes", "Move transposes pass");

}  // namespace TF
}  // namespace mlir
