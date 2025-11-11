/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

// This pass fuses Q and DQ ops and creates quantized kernels.

#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_interface.h.inc"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/lower_quant_annotations_helper.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_FUSEQDQPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

enum QuantizationTrait { kFullQuantization, kDynamicRangeQuantization };

enum class OpQuantizationType { kSRQ, kDRQ, kWeightOnly, kUnsupported };

LogicalResult IsDrqTensor(mlir::Value value, mlir::Value& fq_input) {
  if (auto composite_op = llvm::dyn_cast_or_null<stablehlo::CompositeOp>(
          value.getDefiningOp())) {
    if (IsDrqFakeQuant(composite_op)) {
      fq_input = composite_op.getOperand(0);
      return success();
    }
  }
  return failure();
}

LogicalResult HasDQParent(mlir::Value value, mlir::Value& dq_input) {
  if (auto dq_op =
          llvm::dyn_cast_or_null<DequantizeOp>(value.getDefiningOp())) {
    dq_input = dq_op.getOperand();
    return success();
  }
  return failure();
}

bool IsQuantizableOp(mlir::Operation* op) {
  if (op->hasTrait<OpTrait::TFL::QuantizableResult>()) {
    return true;
  }

  auto custom_op = llvm::dyn_cast_or_null<TFL::CustomTfOp>(op);
  if (!custom_op) {
    return false;
  }

  auto quant_trait = custom_op->getAttrOfType<StringAttr>("_tfl_quant_trait");
  return quant_trait && quant_trait.getValue() == "fully_quantizable";
}

OpQuantizationType GetOpQuantizationType(mlir::Operation* op) {
  // The assumption here is that the op has at least one DQ operand since the
  // pattern's root is that.

  static const absl::NoDestructor<absl::flat_hash_set<std::string>>
      kDrqOpsWithNoDrqInput({"tfl.embedding_lookup"});

  // "return" is not going to be quantized
  if (op->hasTrait<OpTrait::IsTerminator>()) {
    return OpQuantizationType::kUnsupported;
  }

  // Indicates if an input which is not an FQ is seen.
  bool non_fq_float_input_seen = false;
  mlir::Value fq_input, dq_input;
  for (auto operand : op->getOperands()) {
    if (IsDrqTensor(operand, fq_input).succeeded()) {
      // As soon as a DRQ tensor is encountered, the op is DRQ.
      return OpQuantizationType::kDRQ;
    }

    if (HasDQParent(operand, dq_input).succeeded()) {
      // Operands with QDQ can not specify the quantization type.
      continue;
    }

    if (kDrqOpsWithNoDrqInput->contains(op->getName().getStringRef().str())) {
      return OpQuantizationType::kDRQ;
    }

    auto element_type = getElementTypeOrSelf(operand.getType());

    // Ignore non-f32 tensors when determining the quantization type.
    // Examples:
    //  - i32 operands are generally index tensors (e.g. in transpose
    // permutation)
    //  - bool operands can be the `condition` operand in a select_v2 op.
    if (element_type.isF32()) {
      non_fq_float_input_seen = true;
    }
  }
  if (non_fq_float_input_seen) {
    return OpQuantizationType::kWeightOnly;
  }

  for (auto result : op->getResults()) {
    // This check is required othwerwise ops would rematch even after their
    // uses are changed to the quantized version and before they're erased due
    // to being trivially dead.
    // If a result is not used, it doesn't affect the quantization type.
    if (result.use_empty()) {
      return OpQuantizationType::kUnsupported;
    }
    for (auto user : result.getUsers()) {
      if (!llvm::dyn_cast_or_null<QuantizeOp>(user)) {
        return OpQuantizationType::kUnsupported;
      }
    }
  }

  // SRQ kernels need to have quantizable results.
  // Note that this check is required since this can not be fully controlled by
  // user annotations due to the fact that propagation can get annotations
  // around an op that is not quantizable and this check ensures the op is not
  // incorrectly SRQ quantized.
  if (!IsQuantizableOp(op)) {
    return OpQuantizationType::kUnsupported;
  }

  return OpQuantizationType::kSRQ;
}

// Returns a list of operations that consume the result of the given
// DequantizeOp. Returns an empty list if the DequantizeOp has more than one
// result.
SmallVector<mlir::Operation*, 4> GetQuantizingOps(mlir::TFL::DequantizeOp op) {
  llvm::SmallVector<mlir::Operation*, 4> quantizing_ops;
  if (op->getNumResults() != 1) {
    return quantizing_ops;
  }
  auto users = op->getResult(0).getUsers();
  quantizing_ops.append(users.begin(), users.end());
  return quantizing_ops;
}

// Populates the inputs vector with the quantized inputs for the given
// quantizing_op. The inputs are determined based on the op_quant_type. Returns
// failure if an unsupported operand is encountered.
//
// Parameters:
//  quantizing_op: The operation whose inputs are being processed.
//  op_quant_type: The quantization type of the operation.
//  rewriter: The pattern rewriter.
//  inputs: The vector to populate with the quantized inputs.
//  updated: A boolean flag that is set to true if any input is updated.
LogicalResult GetQuantizedInputs(mlir::Operation* quantizing_op,
                                 OpQuantizationType op_quant_type,
                                 PatternRewriter& rewriter,
                                 SmallVector<mlir::Value, 4>& inputs,
                                 bool& updated) {
  inputs.reserve(quantizing_op->getNumOperands());
  for (auto operand : quantizing_op->getOperands()) {
    Type operand_type = operand.getType();

    if (mlir::Value dq_input; HasDQParent(operand, dq_input).succeeded()) {
      if (op_quant_type == OpQuantizationType::kWeightOnly) {
        inputs.push_back(operand);
      } else {
        updated = true;
        inputs.push_back(dq_input);
      }
    } else if (mlir::Value fq_input;
               IsDrqTensor(operand, fq_input).succeeded()) {
      updated = true;
      inputs.push_back(fq_input);
    } else if (auto ele_type = getElementTypeOrSelf(operand_type);
               ele_type.isF32() || ele_type.isInteger(32) ||
               ele_type.isInteger(64) || ele_type.isInteger(1) ||
               mlir::isa<NoneType>(ele_type)) {
      // If it's F32 (non-weight-only and non-drq) or I32 or bool, just
      // directly add the input.
      inputs.push_back(operand);
    } else {
      return rewriter.notifyMatchFailure(
          quantizing_op,
          "has unsupported operand received during quantization");
    }
  }
  return success();
}

// Populates the output_types vector with the quantized output types for the
// given quantizing_op. The output types are determined based on the
// op_quant_type. Also populates the outputs_replaced map with the values to be
// replaced. Returns failure if an unsupported output type is encountered.
//
// Parameters:
//  quantizing_op: The operation whose output types are being determined.
//  op_quant_type: The quantization type of the operation.
//  rewriter: The pattern rewriter.
//  outputs_replaced: A map to populate with values to be replaced. The key is
//    the value to be replaced, and the value is a pair containing the result
//    index and the target QuantizeOp (if any).
//  output_types: The vector to populate with the quantized output types.
//  updated: A boolean flag that is set to true if any output type is updated.
LogicalResult GetQuantizedOutputTypes(
    mlir::Operation* quantizing_op, OpQuantizationType op_quant_type,
    PatternRewriter& rewriter,
    llvm::SmallDenseMap<mlir::Value, std::pair<int, mlir::Operation*>>&
        outputs_replaced,
    SmallVector<Type, 4>& output_types, bool& updated) {
  output_types.reserve(quantizing_op->getNumResults());
  for (const auto& enumerated_result :
       llvm::enumerate(quantizing_op->getResults())) {
    mlir::Value result = enumerated_result.value();
    Type result_type = result.getType();
    Type result_ele_type = getElementTypeOrSelf(result_type);

    if (op_quant_type == OpQuantizationType::kSRQ) {
      int num_observed_qs = 0;
      for (auto user : result.getUsers()) {
        auto user_q_op = llvm::dyn_cast_or_null<QuantizeOp>(user);
        if (!user_q_op) {
          quantizing_op->emitError(
              "SRQ quantized op result should only be used by QuantizeOps.");
          return failure();
        }
        updated = true;
        if (user_q_op->hasAttr("propagated")) {
          outputs_replaced.insert(
              {user_q_op.getInput(), {enumerated_result.index(), user_q_op}});
        } else {
          CHECK(num_observed_qs == 0)
              << "There must be only one observed scale for a tensor.";
          outputs_replaced.insert(
              {user_q_op.getOutput(), {enumerated_result.index(), nullptr}});
          output_types.push_back(user_q_op.getType());
          num_observed_qs++;
        }
      }
      if (num_observed_qs == 0) {
        CHECK(std::distance(result.use_begin(), result.use_end()) == 1)
            << "if there are no observed scales, there must be only one "
               "output annotation that is propagated from the input.";
        auto q_user =
            dyn_cast_or_null<QuantizeOp>(*result.user_begin()).getOutput();
        output_types.push_back(q_user.getType());
      }
    } else if (result_ele_type.isF32() || mlir::isa<NoneType>(result_type)) {
      outputs_replaced.insert({result, {enumerated_result.index(), nullptr}});
      output_types.push_back(result.getType());
    } else {
      return rewriter.notifyMatchFailure(
          quantizing_op,
          "is a fake quantized op with an output that is not float32.");
    }
  }
  return success();
}

// Creates a new quantized operation based on the given quantizing_op.
// The new operation will have the given inputs and output_types.
mlir::Operation* CreateQuantizedOp(mlir::Operation* quantizing_op,
                                   const SmallVector<mlir::Value, 4>& inputs,
                                   const SmallVector<Type, 4>& output_types,
                                   PatternRewriter& rewriter) {
  rewriter.setInsertionPointAfter(quantizing_op);
  OperationState new_state(quantizing_op->getLoc(),
                           quantizing_op->getName().getStringRef(), inputs,
                           output_types, quantizing_op->getAttrs());
  for (int i = 0; i < quantizing_op->getNumRegions(); ++i) {
    new_state.addRegion();
  }
  mlir::Operation* quantized_op = rewriter.create(new_state);
  if (quantizing_op->getNumRegions() != 0) {
    for (const auto& indexed_regions :
         llvm::enumerate(quantizing_op->getRegions())) {
      Region& target_region = quantized_op->getRegion(indexed_regions.index());
      IRMapping mapping;
      indexed_regions.value().cloneInto(&target_region, mapping);
    }
  }
  return quantized_op;
}

// Replaces the uses of the old values in outputs_replaced with the
// corresponding results of the new quantized_op.
void ReplaceUses(
    const llvm::SmallDenseMap<mlir::Value, std::pair<int, mlir::Operation*>>&
        outputs_replaced,
    mlir::Operation* quantized_op, PatternRewriter& rewriter) {
  for (auto output : outputs_replaced) {
    mlir::Value replaced_value = output.getFirst();
    mlir::Operation* target_q_op = output.getSecond().second;
    int result_index = output.getSecond().first;
    if (target_q_op) {
      rewriter.replaceUsesWithIf(
          replaced_value, quantized_op->getResult(result_index),
          [&](OpOperand& use) { return use.getOwner() == target_q_op; });
    } else {
      rewriter.replaceAllUsesWith(replaced_value,
                                  quantized_op->getResult(result_index));
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

class RemoveUnusedFQ : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const final {
    if (IsDrqFakeQuant(op) && op->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "is not a drq fake quant op with no uses.");
  }
};

// Pushes a drq fake quant op forward through a pad op.
// This is to allow DRQ FQ to be fused into the DRQ op.
// drq_fake_quant(input) -> pad -> output
// becomes
// input -> pad -> drq_fake_quant -> output
class PushForwardDrqFQ : public OpRewritePattern<stablehlo::CompositeOp> {
 public:
  using OpRewritePattern<stablehlo::CompositeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CompositeOp drq_fq_op,
                                PatternRewriter& rewriter) const final {
    if (!IsDrqFakeQuant(drq_fq_op)) {
      return rewriter.notifyMatchFailure(drq_fq_op,
                                         "is not a drq fake quant op.");
    }

    if (!drq_fq_op.getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(
          drq_fq_op, "drq fake quant op does not have one use.");
    }
    auto pad_op =
        llvm::dyn_cast<TFL::PadOp>(*drq_fq_op.getResult(0).user_begin());
    if (!pad_op) {
      return rewriter.notifyMatchFailure(drq_fq_op,
                                         "user is not a tfl.pad op.");
    }

    // The input to the new pad op is the float input to the drq fake quant op.
    mlir::Value float_input =
        drq_fq_op.getOperand(drq_fq_op.getNumOperands() - 1);

    // Create a new pad op.
    auto new_pad_op = rewriter.create<TFL::PadOp>(
        pad_op.getLoc(), pad_op.getType(), float_input, pad_op.getPadding());

    // Create a new drq fake quant op.
    // Operands are the same, except for the last one.
    SmallVector<mlir::Value> new_drq_operands;
    for (mlir::Value operand : drq_fq_op.getOperands().drop_back()) {
      new_drq_operands.push_back(operand);
    }
    new_drq_operands.push_back(new_pad_op.getResult());

    auto new_drq_fq_op = rewriter.create<stablehlo::CompositeOp>(
        drq_fq_op.getLoc(), pad_op.getType(), new_drq_operands,
        drq_fq_op->getAttrs());

    rewriter.replaceOp(pad_op, new_drq_fq_op.getResult(0));
    return success();
  }
};

class FuseQDQ : public OpRewritePattern<mlir::TFL::DequantizeOp> {
  using OpRewritePattern<mlir::TFL::DequantizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::TFL::DequantizeOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<mlir::Operation*, 4> quantizing_ops = GetQuantizingOps(op);
    if (quantizing_ops.empty() && op->getNumResults() == 1) {
      return failure();
    }

    bool updated = false;

    // Rewrite the floating-point ops to the quantized version, by fusing
    // preceding dequantize ops and succeding quantize ops.
    for (mlir::Operation* quantizing_op : quantizing_ops) {
      auto op_quant_type = GetOpQuantizationType(quantizing_op);

      if (op_quant_type == OpQuantizationType::kUnsupported) {
        return rewriter.notifyMatchFailure(quantizing_op,
                                           "has unsupported quantization type");
      }

      SmallVector<mlir::Value, 4> inputs;
      if (failed(GetQuantizedInputs(quantizing_op, op_quant_type, rewriter,
                                    inputs, updated))) {
        return failure();
      }

      llvm::SmallDenseMap<mlir::Value, std::pair<int, mlir::Operation*>>
          outputs_replaced;
      SmallVector<Type, 4> output_types;
      if (failed(GetQuantizedOutputTypes(quantizing_op, op_quant_type, rewriter,
                                         outputs_replaced, output_types,
                                         updated))) {
        return failure();
      }

      if (!updated) {
        return rewriter.notifyMatchFailure(
            op, "has no further opportunities to fuse Q's and DQ's.");
      }

      mlir::Operation* quantized_op =
          CreateQuantizedOp(quantizing_op, inputs, output_types, rewriter);

      ReplaceUses(outputs_replaced, quantized_op, rewriter);
    }
    return success();
  }
};

// We need control over when this happens and so this cannot happen as a folder.
// Some optimizations like fusing a mul following a conv/FC into rhs, change the
// scale of rhs. So, if we fold Q into rhs early, we'll need a requant later
// which is losing information twice.
class QuantizeConstPattern : public OpRewritePattern<QuantizeOp> {
 public:
  // Does not take ownership of context, which must not be null and must outlive
  // this pattern.
  explicit QuantizeConstPattern(MLIRContext* context)
      : OpRewritePattern<QuantizeOp>(context) {}
  LogicalResult matchAndRewrite(QuantizeOp op,
                                PatternRewriter& rewriter) const override {
    DenseFPElementsAttr attr;
    if (matchPattern(op.getInput(), m_Constant(&attr))) {
      auto qtype = op.getQtypeAttr();
      Attribute quantized_attr = mlir::TFL::Quantize(attr, qtype.getValue());
      if (quantized_attr) {
        auto qconst_op =
            rewriter.create<QConstOp>(op.getLoc(), qtype, quantized_attr);
        if (auto volatile_attr = op->getAttr(mlir::TFL::kVolatileOpAttrName)) {
          qconst_op->setAttr(mlir::TFL::kVolatileOpAttrName, volatile_attr);
        }
        op.replaceAllUsesWith(qconst_op.getOutput());
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct FuseQDQPass : public impl::FuseQDQPassBase<FuseQDQPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseQDQPass)

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/lite/transforms/quantization/generated_strict_quantize.inc"

void FuseQDQPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<FuseQDQ, PushForwardDrqFQ, RemoveUnusedFQ, QuantizeConstPattern,
               FuseDqQToRequant, FuseQQToRequant, RemoveNoOpQ>(ctx);

  // Configure the greedy pattern rewrite driver.
  GreedyRewriteConfig greedy_config;

  if (failed(
          applyPatternsGreedily(module, std::move(patterns), greedy_config))) {
    module.emitError("Failed to apply FuseQDQPass patterns.");
    signalPassFailure();
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<mlir::ModuleOp>> CreateFuseQDQPass() {
  return std::make_unique<FuseQDQPass>();
}

}  // namespace TFL
}  // namespace mlir
