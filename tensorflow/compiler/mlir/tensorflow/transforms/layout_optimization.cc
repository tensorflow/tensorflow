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

#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#define DEBUG_TYPE "tf-layout-optimization"

namespace mlir {
namespace TF {

namespace {

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

}  // namespace

static PassRegistration<LayoutAssignmentPass> pass("tf-layout-assignment",
                                                   "Layout assignment pass");

}  // namespace TF
}  // namespace mlir
