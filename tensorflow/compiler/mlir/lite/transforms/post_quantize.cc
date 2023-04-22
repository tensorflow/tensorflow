/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass applies some clean up steps after quantization.

#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

//===----------------------------------------------------------------------===//
// The post-quantize Passes.
//
namespace mlir {
namespace TFL {
namespace {

// Applies all the clean up steps after quantization.
class PostQuantizePass : public PassWrapper<PostQuantizePass, FunctionPass> {
 public:
  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit PostQuantizePass() : emit_quant_adaptor_ops_(false) {}

  // Constructor used by manually creating the pass.
  explicit PostQuantizePass(bool emit_quant_adaptor_ops)
      : emit_quant_adaptor_ops_(emit_quant_adaptor_ops) {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-post-quantize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply post quantization clean up after quantization";
  }

  void runOnFunction() override;

 private:
  // Set this flag to true if the inputs and outputs are in floating point. The
  // quant adaptor ops convert them to fixed point values (i.e. quantize) before
  // feeding them to the model and convert them back to floating point
  // (i.e. dequantize) as the output.
  bool emit_quant_adaptor_ops_;
};

// Cleans up unnecessary QDQ pattern for input/output ops.
class PostQuantizeRemoveQDQPass
    : public PassWrapper<PostQuantizeRemoveQDQPass, FunctionPass> {
 public:
  // Constructor used by the PassRegistration. This will remove QDQ ops.
  explicit PostQuantizeRemoveQDQPass() {}

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-post-quantize-remove-qdq";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Remove qdq from input and output nodes after quantization";
  }

  void runOnFunction() override;
};

// TODO(fengliuai): migrate to use modify_io_nodes pass.
void RemoveQuantizationAdaptorOps(FuncOp func) {
  mlir::OpBuilder builder(func.getBody());
  auto& bb = func.front();

  int num_args = bb.getNumArguments();
  llvm::SmallVector<Type, 4> input_types;
  input_types.reserve(num_args);
  // Edit the block arguments and create the new input ops in place to replace
  // the old input ops and quantize ops.
  for (int i = 0; i != num_args; ++i) {
    // Previous loop iteration may invalidate the insertion point so we have to
    // reset insertion point each iteration.
    builder.setInsertionPointToStart(&bb);

    // In each iteration, a new argument is appended to the end of the list
    // and the current argument is erased, so here we always process the first
    // argument in the list.
    auto arg = bb.getArgument(0);

    auto remove_quantize_op = [&](QuantizeOp quantize_op) {
      auto quantize_output = quantize_op.output();
      auto quantize_type = quantize_output.getType();
      input_types.push_back(quantize_type);
      auto new_arg = bb.addArgument(quantize_type);
      quantize_output.replaceAllUsesWith(new_arg);
      quantize_op.erase();
      arg.dropAllUses();
      bb.eraseArgument(0);
    };

    // This is looking for a pattern: arg -> tfl.quantize
    if (arg.hasOneUse() && llvm::isa<QuantizeOp>(*arg.user_begin())) {
      auto quantize_op = llvm::cast<QuantizeOp>(*arg.user_begin());
      remove_quantize_op(quantize_op);
      continue;
    }

    // Make a copy of current argument and append it to the end of the list if
    // the pattern isn't found.
    Type arg_type = arg.getType();
    input_types.push_back(arg_type);
    auto new_arg = bb.addArgument(arg_type);
    arg.replaceAllUsesWith(new_arg);
    arg.dropAllUses();
    bb.eraseArgument(0);
  }

  // Edit the return ops and remove the dequantize ops in place.
  auto* terminator = bb.getTerminator();
  int num_return_operands = terminator->getNumOperands();
  llvm::SmallVector<Type, 4> output_types;
  output_types.reserve(num_return_operands);
  for (int i = 0; i != num_return_operands; ++i) {
    auto returned_value = terminator->getOperand(i);
    Operation* returned_op = returned_value.getDefiningOp();
    if (returned_op && returned_op->hasOneUse() &&
        llvm::isa<DequantizeOp>(returned_op)) {
      auto dequantize_op = llvm::cast<DequantizeOp>(returned_op);
      Value dequantized_result = dequantize_op.input();
      output_types.push_back(dequantized_result.getType());
      terminator->setOperand(i, dequantized_result);
      returned_op->erase();
    } else {
      output_types.push_back(returned_value.getType());
    }
  }
  auto new_func_type = builder.getFunctionType(input_types, output_types);
  func.setType(new_func_type);
}

enum RemoveVolatileOpsType {
  // Remove all volatile quant-dequant ops.
  kPreserveNone,
  // Preserve volatile quant-dequants for input and output ops.
  kPreserveInputsAndOutputs,
};

// Remove the back-to-back quantize and dequantize ops with volatile attribute.
template <RemoveVolatileOpsType remove_volatile_ops_type>
struct RemoveVolatileOps : public OpRewritePattern<DequantizeOp> {
  explicit RemoveVolatileOps(MLIRContext* context)
      : OpRewritePattern<DequantizeOp>(context, 1) {}

  LogicalResult matchAndRewrite(DequantizeOp op,
                                PatternRewriter& rewriter) const override {
    auto input_op = op.input().getDefiningOp();
    if (auto q = llvm::dyn_cast_or_null<QuantizeOp>(input_op)) {
      if (!q->getAttr(mlir::quant::kVolatileOpAttrName)) return failure();

      if (remove_volatile_ops_type == kPreserveInputsAndOutputs) {
        // Don't remove leading and tailing QDQ for PQT workflow, so the io
        // modifying lib can work correctly.
        if (!q.input().getDefiningOp()) return failure();
        if (op->hasOneUse() &&
            op->user_begin()->hasTrait<OpTrait::IsTerminator>())
          return failure();
      }
      // If the quantize op is a requantize op, it is being used in other scale
      // adjustments and should be kept. Instead, moving dequantize op before
      // the requantize op to remove the unnecessary requantize op.
      if (auto qtype = quant::QuantizedType::getQuantizedElementType(
              q.input().getType())) {
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<DequantizeOp>(op, op.output().getType(),
                                                  q.input());
        return success();
      }

      op.replaceAllUsesWith(q.input());
      return success();
    }
    return failure();
  }
};

// Removes operations with side effect (i.e. LSTM, SVDF) that have dangling
// output.
template <typename OpTy>
struct PruneUnusedOpsWithSideEffect : public OpRewritePattern<OpTy> {
 public:
  explicit PruneUnusedOpsWithSideEffect(MLIRContext* context)
      : OpRewritePattern<OpTy>(context) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter& rewriter) const override {
    if (op.getOperation()->template hasTrait<OpTrait::IsTerminator>()) {
      return failure();
    }
    for (auto result : op.getOperation()->getOpResults()) {
      if (!result.use_empty()) {
        return failure();
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_post_quantize.inc"

void PostQuantizePass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.insert<quant::FoldTrivalRequantizeOp<QuantizeOp>>(ctx);
  patterns.insert<PruneUnusedOpsWithSideEffect<TFL::LSTMOp>>(ctx);
  patterns
      .insert<PruneUnusedOpsWithSideEffect<TFL::UnidirectionalSequenceLSTMOp>>(
          ctx);
  patterns.insert<PruneUnusedOpsWithSideEffect<TFL::SVDFOp>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  if (!emit_quant_adaptor_ops_) {
    RemoveQuantizationAdaptorOps(getFunction());
  }

  OwningRewritePatternList phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.insert<quant::FoldTrivalRequantizeOp<QuantizeOp>,
                          RemoveVolatileOps<kPreserveInputsAndOutputs>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_2_patterns));
}

void PostQuantizeRemoveQDQPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.insert<RemoveVolatileOps<kPreserveNone>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops) {
  return std::make_unique<PostQuantizePass>(emit_quant_adaptor_ops);
}

// Creates an instance of the TensorFlow Lite dialect PostQuantizeRemoveQDQ
// pass.
std::unique_ptr<OperationPass<FuncOp>> CreatePostQuantizeRemoveQDQPass() {
  return std::make_unique<PostQuantizeRemoveQDQPass>();
}

static PassRegistration<PostQuantizePass> pass;

static PassRegistration<PostQuantizeRemoveQDQPass> remove_qdq_pass;

}  // namespace TFL
}  // namespace mlir
