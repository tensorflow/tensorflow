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
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir {
namespace odml {
namespace {

// TODO - b/330337238: Surface these to other files when needed.
constexpr llvm::StringLiteral kCompositeNamespace = "odml.internal";
constexpr llvm::StringLiteral kGelu = "gelu";

std::string MakeCompositeName(llvm::StringRef op_name) {
  return (kCompositeNamespace + "." + op_name).str();
}

#define GEN_PASS_DEF_OUTLINECOMPOSITESPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/odml_converter/passes.h.inc"

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

// Builds a reference implementation of non-approximate GELU.
func::FuncOp BuildGELUDecomposition(RankedTensorType type,
                                    PatternRewriter& rewriter,
                                    Block* insertion_point) {
  rewriter.setInsertionPointToStart(insertion_point);

  auto ftype = FunctionType::get(rewriter.getContext(), {type}, {type});
  auto name = rewriter.getStringAttr("gelu_decomp");
  func::FuncOp new_func = rewriter.create<func::FuncOp>(
      insertion_point->front().getLoc(), name, ftype);
  new_func.setPrivate();
  new_func.addEntryBlock();
  rewriter.setInsertionPointToStart(&new_func.getBody().front());

  auto one_val = DenseElementsAttr::get(type, kOne);
  auto one_cst =
      rewriter.create<stablehlo::ConstantOp>(rewriter.getUnknownLoc(), one_val);

  auto half_val = DenseElementsAttr::get(type, kHalf);
  auto half_cst =
      rewriter.create<stablehlo::ConstantOp>(one_cst.getLoc(), half_val);

  auto one_over_root2_val = DenseElementsAttr::get(type, kOneOverRoot2);
  auto one_over_root2_cst = rewriter.create<stablehlo::ConstantOp>(
      half_cst.getLoc(), one_over_root2_val);

  auto mul_op = rewriter.create<stablehlo::MulOp>(one_over_root2_cst.getLoc(),
                                                  new_func.getArguments()[0],
                                                  one_over_root2_cst);
  auto erf_op = rewriter.create<chlo::ErfOp>(mul_op.getLoc(), mul_op);
  auto add_op =
      rewriter.create<stablehlo::AddOp>(erf_op.getLoc(), erf_op, one_cst);
  auto lhs_mul_op = rewriter.create<stablehlo::MulOp>(
      half_cst.getLoc(), new_func.getArguments()[0], half_cst);
  auto output_mul_op = rewriter.create<stablehlo::MulOp>(lhs_mul_op.getLoc(),
                                                         lhs_mul_op, add_op);

  rewriter.create<func::ReturnOp>(output_mul_op.getLoc(),
                                  output_mul_op.getResult());
  rewriter.clearInsertionPoint();
  return new_func;
}

// Outlines non-approximate GELU into a stablehlo composite.
//
//    -> mul 1/sqrt(2) -> erf -> add 1 ->
// in                                    mul
//    ---------> mul 0.5 --------------->
//
// This pattern assumes all binary ewise ops with one constant argument
// have that constant argument as the second operand. It works by
// identifying `erf` ops and validate the structure around them.
class OutlineGELU : public RewritePattern {
 public:
  explicit OutlineGELU(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
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

    auto lhs_mul =
        llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_user_user_input);
    if (!lhs_mul) return failure();

    auto output_mul = llvm::dyn_cast_or_null<stablehlo::MulOp>(erf_user_user);
    if (!output_mul) return failure();

    auto rhs_add = llvm::dyn_cast_or_null<stablehlo::AddOp>(erf_user);
    if (!rhs_add) return failure();

    if (!HasSplatArg(rhs_add, kOne, 1)) return failure();
    if (!HasSplatArg(lhs_mul, kHalf, 1)) return failure();
    if (!HasSplatArg(rhs_mul, kOneOverRoot2, 1)) return failure();

    // Build a function to serve as the GELU decomposition in the
    // shlo composite op.
    auto root = op->getParentOfType<ModuleOp>();
    auto func = BuildGELUDecomposition(
        rhs_add.getType().cast<RankedTensorType>(), rewriter, root.getBody());

    SymbolTable table(root);
    (void)table.renameToUnique(func, {});

    rewriter.setInsertionPointAfter(output_mul);
    auto composite_attrs = rewriter.getDictionaryAttr(
        {rewriter.getNamedAttr("approx", rewriter.getBoolAttr(false))});
    auto composite_op = rewriter.create<stablehlo::CompositeOp>(
        output_mul.getLoc(), func.getResultTypes()[0],
        SmallVector<Value>{erf_input->getOperand(0)}, MakeCompositeName(kGelu),
        composite_attrs, func.getSymName());
    rewriter.replaceAllOpUsesWith(output_mul, composite_op);
    // Note these must be erased in reverse topo order to avoid
    // failing in debug mode.
    rewriter.eraseOp(output_mul);
    rewriter.eraseOp(rhs_add);
    rewriter.eraseOp(op);
    rewriter.eraseOp(lhs_mul);
    rewriter.eraseOp(rhs_mul);

    return success();
  }
};

class OutlineCompositesPass
    : public impl::OutlineCompositesPassBase<OutlineCompositesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OutlineCompositesPass)

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<OutlineGELU>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateOutlineCompositesPass() {
  return std::make_unique<OutlineCompositesPass>();
}

}  // namespace odml
}  // namespace mlir
