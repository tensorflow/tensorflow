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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"

namespace mlir {
namespace quant {
namespace {

class LiftQuantizableSpotsAsFunctionsPass
    : public PassWrapper<LiftQuantizableSpotsAsFunctionsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LiftQuantizableSpotsAsFunctionsPass)

  LiftQuantizableSpotsAsFunctionsPass() = default;

  explicit LiftQuantizableSpotsAsFunctionsPass(OpSet op_set,
                                               bool enable_two_input_tensors) {
    op_set_ = op_set;
    enable_two_input_tensors_ = enable_two_input_tensors;
  }

  LiftQuantizableSpotsAsFunctionsPass(
      const LiftQuantizableSpotsAsFunctionsPass& other) {
    op_set_ = other.op_set_;
    enable_two_input_tensors_ = other.enable_two_input_tensors_;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-lift-quantizable-spots-as-functions";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Replace quantization candidates with composite functions into the "
           "module";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;

 private:
  Option<OpSet> op_set_{
      *this, "target-opset", llvm::cl::init(OpSet::TF),
      llvm::cl::desc("Choose target opset."),
      llvm::cl::values(
          clEnumValN(OpSet::TF, "TF",
                     "Uses TF ops that mimic quantization behavior"),
          clEnumValN(OpSet::XLA, "XLA", "Uses TF XLA ops"),
          clEnumValN(OpSet::UNIFORM_QUANTIZED, "UNIFORM_QUANTIZED",
                     "Uses TF Uniform Quantized ops"))};

  bool enable_two_input_tensors_{false};
};

class CheckQuantizableOps
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit CheckQuantizableOps(MLIRContext* context, OpSet op_set,
                               bool enable_two_input_tensors)
      : OpRewritePattern<TF::PartitionedCallOp>(context),
        op_set_(op_set),
        enable_two_input_tensors_(enable_two_input_tensors) {}

 private:
  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
    StringRef function_name =
        call_op.getFAttr().cast<FlatSymbolRefAttr>().getValue();
    if (!function_name.startswith("composite_") ||
        !call_op->hasAttr(kQuantTraitAttrName)) {
      return failure();
    }

    absl::Status check_status;
    // Skip quantization for read-only ops as only weight-only is supported.
    if (function_name.contains("gather")) {
      check_status.Update(absl::InternalError("Weight-only op is skipped."));
    }

    if (op_set_ == OpSet::XLA) {
      check_status.Update(checkQuantizableOpsForXla(call_op, function_name,
                                                    enable_two_input_tensors_));
    }

    // Only the composite functions with f32 inputs are quantizable.
    if (call_op.getResults().size() == 1 && !call_op->getResult(0)
                                                 .getType()
                                                 .cast<ShapedType>()
                                                 .getElementType()
                                                 .isF32()) {
      check_status.Update(absl::InternalError(
          "Composite functions for quantization should be f32 type."));
    }

    // The OK status means this op is quantizable. Return failure since the
    // pattern doesn't rewrite anything yet.
    if (check_status.ok()) return failure();
    call_op->removeAttr(kQuantTraitAttrName);
    removeAttrMapAttribute(call_op, function_name, check_status.message());
    return success();
  }

  absl::Status checkQuantizableOpsForXla(TF::PartitionedCallOp call_op,
                                         StringRef function_name,
                                         bool enable_two_input_tensors) const {
    // Disable quantization for the DepthwiseConv since it has no benefits in
    // the XLA opset.
    if (function_name.contains("depthwise_conv2d")) {
      return absl::InternalError(
          "DepthwiseConv2D doesn't get any benefit of quantization in XLA.");
    } else if (function_name.contains("conv2d")) {
      // For Conv2D, the channel dimension must be static to calculate the
      // feature group count.
      if (!HasStaticShapeAtDims(call_op->getOperand(0), /*dims=*/3)) {
        return absl::InternalError(
            "The channel dimension of Conv2D is required to be static.");
      }
    } else if (function_name.contains("conv3d")) {
      // For Conv3D, the channel dimension must be static to calculate the
      // feature group count.
      if (!HasStaticShapeAtDims(call_op->getOperand(0), /*dims=*/4)) {
        return absl::InternalError(
            "The channel dimension of Conv3D is required to be static.");
      }
    } else if (function_name.contains("batch_matmul")) {
      // For BatchMatMul, the input must be ranked.
      auto shaped_type =
          call_op->getOperand(0).getType().dyn_cast<ShapedType>();
      if (!shaped_type || !shaped_type.hasRank()) {
        return absl::InternalError("The input of BatchMatMul must have rank.");
      }
    }

    std::unique_ptr<OpQuantSpec> spec = GetTFOpQuantSpec(call_op);
    for (auto iter : spec->coeff_op_quant_dim) {
      Operation* preceding_op = call_op.getOperand(iter.first).getDefiningOp();
      // The XLA opset only supports constant filter/weight at the moment.
      bool is_weight_constant =
          preceding_op && preceding_op->hasTrait<OpTrait::ConstantLike>();

      // There might be q/dq ops after the filter/weight.
      if (auto dq_op = llvm::dyn_cast_or_null<quantfork::DequantizeCastOp>(
              preceding_op)) {
        if (auto q_op = llvm::dyn_cast_or_null<quantfork::QuantizeCastOp>(
                dq_op.getArg().getDefiningOp())) {
          Operation* q_op_input = q_op.getArg().getDefiningOp();
          is_weight_constant =
              q_op_input && q_op_input->hasTrait<OpTrait::ConstantLike>();
        }
      }

      if (!is_weight_constant) {
        if (!enable_two_input_tensors || (!function_name.contains("matmul") &&
                                          !function_name.contains("einsum"))) {
          return absl::InternalError(
              "Non-constant weights are not supported at the moment,"
              " except matmul and einsum.");
        }
      }
    }
    return absl::OkStatus();
  }

  void removeAttrMapAttribute(TF::PartitionedCallOp call_op,
                              StringRef function_name,
                              StringRef error_message) const {
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);
    mlir::func::FuncOp composite_func =
        dyn_cast<func::FuncOp>(symbol_table.lookup(function_name));
    if (!composite_func) return;

    composite_func.walk([&](Operation* op) {
      if (op->hasAttr(kAttrMapAttribute)) {
        op->removeAttr(kAttrMapAttribute);

        std::string log_message;
        llvm::raw_string_ostream log_stream(log_message);
        op->getLoc().print(log_stream);
        log_stream << ": Quantization disabled on this op: ";
        log_stream << error_message << "\n";
        log_stream << "See the current operation:\n";
        op->print(log_stream);
        VLOG(2) << log_message;
      }
    });
  }

  OpSet op_set_;
  bool enable_two_input_tensors_;
};

static PassRegistration<LiftQuantizableSpotsAsFunctionsPass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/lift_quantizable_spots_as_functions.inc"

void LiftQuantizableSpotsAsFunctionsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  ModuleOp module = getOperation();

  populateWithGenerated(patterns);
  patterns.add<CheckQuantizableOps>(ctx, op_set_, enable_two_input_tensors_);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  for (auto func : module.getOps<func::FuncOp>()) {
    if (failed(applyPatternsAndFoldGreedily(func, frozen_patterns))) {
      func.emitError() << "quant-lift-quantizable-spots-as-functions failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateLiftQuantizableSpotsAsFunctionsPass(OpSet target_opset,
                                          bool enable_two_input_tensors) {
  return std::make_unique<LiftQuantizableSpotsAsFunctionsPass>(
      target_opset, enable_two_input_tensors);
}

}  // namespace quant
}  // namespace mlir
