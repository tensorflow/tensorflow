/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/gelu.h"

#include <cmath>
#include <cstdlib>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep

namespace mlir::odml {

constexpr float kOne = 1.0;
const float kOneOverRoot2 = kOne / std::sqrt(2);
constexpr float kHalf = kOne / 2.0;
constexpr float kTolerance = kOne / 1000.0;

// Gets the operation that uses the sole result of given operation
// if there is only one.
Operation* GetUserIfOnlyOne(Operation* op) {
  if (op->getNumResults() != 1) return nullptr;
  auto result = op->getResult(0);
  if (!result.hasOneUse()) return nullptr;
  return (*result.getUses().begin()).getOwner();
}

// Gets operation providing value for the given operand of given operation
// if the given operation is the only user.
Operation* GetInputOpWithOneUse(Operation* op, int opr_num) {
  if (opr_num >= op->getNumOperands()) return nullptr;
  auto opr = op->getOperand(opr_num);
  if (llvm::isa<BlockArgument>(opr)) return nullptr;
  auto* res = opr.getDefiningOp();
  if (!res->hasOneUse()) return nullptr;
  return res;
}

// Checks if the given operand of given operation refers to a splat constant
// with given val.
bool HasSplatArg(Operation* op, float val, int opr_num) {
  auto* cst_input = GetInputOpWithOneUse(op, 1);
  if (!cst_input) return false;
  auto cst_op = llvm::dyn_cast_or_null<stablehlo::ConstantOp>(cst_input);
  if (!cst_op) return false;
  ElementsAttr value = cst_op.getValue();
  if (!value.isSplat()) return false;
  if (!value.getElementType().isF32()) return false;
  return std::abs(value.getSplatValue<float>() - val) < kTolerance;
}

// Determines if the given op is semantically that of the gauss error function.
bool MatchERF(Operation* op) {
  if (auto custom_call = llvm::dyn_cast_or_null<stablehlo::CustomCallOp>(op)) {
    return custom_call.getCallTargetName() == "mhlo.erf";
  }
  return llvm::isa<chlo::ErfOp>(op);
}

LogicalResult LowerGELU::matchAndRewrite(Operation* op,
                                         PatternRewriter& rewriter) const {
  if (!MatchERF(op)) return failure();
  // `add 1`
  auto* erf_user = GetUserIfOnlyOne(op);
  if (!erf_user) return failure();

  // `mul`
  auto* erf_user_user = GetUserIfOnlyOne(erf_user);
  if (!erf_user_user) return failure();

  // `mul 1/sqrt(2)`
  auto* erf_input = GetInputOpWithOneUse(op, 0);
  if (!erf_input) return failure();

  // `mul 0.5`
  auto* erf_user_user_input = GetInputOpWithOneUse(erf_user_user, 0);
  if (!erf_user_user_input) return failure();

  // Check `mul 0.5` and `mul 1/sqrt(2)` refer to the same input.
  if (erf_user_user_input->getOperand(0) != erf_input->getOperand(0)) {
    return failure();
  }

  // Check the structural matches have the correct op type and values.
  auto rhs_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_input);
  if (!rhs_mul) return failure();

  auto lhs_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_user_user_input);
  if (!lhs_mul) return failure();

  auto output_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_user_user);
  if (!output_mul) return failure();

  auto rhs_add = llvm::dyn_cast_or_null<stablehlo::AddOp>(erf_user);
  if (!rhs_add) return failure();

  if (!HasSplatArg(rhs_add, kOne, 1)) return failure();
  if (!HasSplatArg(lhs_mul, kHalf, 1)) return failure();
  if (!HasSplatArg(rhs_mul, kOneOverRoot2, 1)) return failure();

  auto is_approx_attr = rewriter.getBoolAttr(false);
  auto gelu = rewriter.create<TFL::GeluOp>(
      output_mul.getLoc(), output_mul.getResult().getType(),
      erf_input->getOperand(0), is_approx_attr);
  rewriter.replaceAllOpUsesWith(output_mul, gelu);
  // Note these must be erased in reverse topo order to avoid
  // failing in debug mode.
  rewriter.eraseOp(output_mul);
  rewriter.eraseOp(rhs_add);
  rewriter.eraseOp(op);
  rewriter.eraseOp(lhs_mul);
  rewriter.eraseOp(rhs_mul);

  return success();
}

}  // namespace mlir::odml
