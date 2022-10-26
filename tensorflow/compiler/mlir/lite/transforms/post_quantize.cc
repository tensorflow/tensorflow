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

#include <string>
#include <utility>

#include "llvm/Support/Casting.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

//===----------------------------------------------------------------------===//
// The post-quantize Passes.
//
namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_POSTQUANTIZEPASS
#define GEN_PASS_DEF_POSTQUANTIZEREMOVEQDQPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Applies all the clean up steps after quantization.
class PostQuantizePass : public impl::PostQuantizePassBase<PostQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostQuantizePass)

  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit PostQuantizePass() { this->emit_quant_adaptor_ops_ = false; }

  // Constructor used by manually creating the pass.
  explicit PostQuantizePass(bool emit_quant_adaptor_ops,
                            const quant::CustomOpMap& custom_op_map)
      : custom_op_map_(custom_op_map) {
    // Set this flag to true if the inputs and outputs are in floating point.
    // The quant adaptor ops convert them to fixed point values (i.e. quantize)
    // before feeding them to the model and convert them back to floating point
    // (i.e. dequantize) as the output.
    this->emit_quant_adaptor_ops_ = emit_quant_adaptor_ops;
  }

  void runOnOperation() override;

 private:
  quant::CustomOpMap custom_op_map_;
};

// Cleans up unnecessary QDQ pattern for input/output ops.
class PostQuantizeRemoveQDQPass
    : public impl::PostQuantizeRemoveQDQPassBase<PostQuantizeRemoveQDQPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostQuantizeRemoveQDQPass)

  void runOnOperation() override;
};

// TODO(fengliuai): migrate to use modify_io_nodes pass.
void RemoveQuantizationAdaptorOps(func::FuncOp func) {
  mlir::OpBuilder builder(func.getBody());
  auto& bb = func.front();
  auto loc = func.getLoc();

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
      auto quantize_output = quantize_op.getOutput();
      auto quantize_type = quantize_output.getType();
      input_types.push_back(quantize_type);
      auto new_arg = bb.addArgument(quantize_type, loc);
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
    auto new_arg = bb.addArgument(arg_type, loc);
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
      Value dequantized_result = dequantize_op.getInput();
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
    auto input_op = op.getInput().getDefiningOp();
    if (auto q = llvm::dyn_cast_or_null<QuantizeOp>(input_op)) {
      if (!q->getAttr(mlir::quant::kVolatileOpAttrName)) return failure();

      if (remove_volatile_ops_type == kPreserveInputsAndOutputs) {
        // Don't remove leading and trailing QDQ for PTQ workflow, so the io
        // modifying lib can work correctly.
        if (!q.getInput().getDefiningOp()) return failure();
        if (op->hasOneUse() &&
            op->user_begin()->hasTrait<OpTrait::IsTerminator>())
          return failure();
      }
      // If the quantize op is a requantize op, it is being used in other scale
      // adjustments and should be kept. Instead, moving dequantize op before
      // the requantize op to remove the unnecessary requantize op.
      if (auto qtype = quant::QuantizedType::getQuantizedElementType(
              q.getInput().getType())) {
        rewriter.setInsertionPoint(op);
        rewriter.replaceOpWithNewOp<DequantizeOp>(op, op.getOutput().getType(),
                                                  q.getInput());
        return success();
      }

      op.replaceAllUsesWith(q.getInput());
      return success();
    }
    return failure();
  }
};

// Fold the constant quantized Transpose ops.
struct FoldTransposeOp : public OpRewritePattern<TransposeOp> {
  explicit FoldTransposeOp(MLIRContext* context)
      : OpRewritePattern<TransposeOp>(context, 1) {}

  // Computes the permutation of a constant `input_tensor` according to `perm`.
  // The function recursively traverses the dimensions of the output tensor in
  // a row-major order and writes the value in the output tensor into
  // `new_values`.
  void ComputePermutation(ElementsAttr input_tensor, ArrayRef<int32_t> perm,
                          ArrayRef<int64_t> output_shape, int num_dimensions,
                          int output_axis, std::vector<uint64_t>* input_indices,
                          std::vector<Attribute>* new_values) const {
    // Refer to the implementation of `Transpose` function in
    // tensorflow/lite/kernels/internal/reference/reference_ops.h
    assert(output_axis < num_dimensions);
    const int input_axis = perm[output_axis];
    for (int i = 0; i < output_shape[output_axis]; ++i) {
      // Update the input indices on `input_axis`.
      assert(input_axis < input_indices->size());
      input_indices->operator[](input_axis) = static_cast<uint64_t>(i);
      // Write the value from `input_tensor` if it is the last axis or
      // recurse into the next axis.
      const bool is_last_axis = output_axis == num_dimensions - 1;
      if (is_last_axis) {
        new_values->push_back(
            input_tensor.getValues<Attribute>()[*input_indices]);
      } else {
        ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                           output_axis + 1, input_indices, new_values);
      }
    }
  }

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def_op = op.getInput().getDefiningOp();
    auto qconst_op = llvm::dyn_cast_or_null<QConstOp>(def_op);
    if (qconst_op == nullptr) return failure();

    DenseIntElementsAttr perm_tensor;
    if (!matchPattern(op.getPerm(), m_Constant(&perm_tensor))) return failure();

    if (!(getElementTypeOrSelf(op.getOutput().getType()))
             .isa<quant::UniformQuantizedType>())
      return failure();

    ElementsAttr input_tensor = qconst_op.getValue();

    assert(perm_tensor.getType().getRank() == 1);
    const int num_dimensions = input_tensor.getType().getRank();
    assert(perm_tensor.getType().getNumElements() == num_dimensions);

    ArrayRef<int64_t> input_shape = input_tensor.getType().getShape();
    auto output_type = op.getOutput().getType().cast<ShapedType>();

    SmallVector<int32_t, 4> perm;
    SmallVector<int64_t, 4> output_shape;
    for (int i = 0; i < num_dimensions; ++i) {
      perm.push_back(perm_tensor.getValues<IntegerAttr>()[i].getInt());
      output_shape.push_back(input_shape[perm[i]]);

      // Check that the derived output shape matches the static shape.
      assert(!output_type.hasStaticShape() ||
             output_type.getShape()[i] == output_shape[i]);
    }

    std::vector<Attribute> new_values;
    new_values.reserve(input_tensor.getType().getNumElements());
    std::vector<uint64_t> input_indices(num_dimensions);
    ComputePermutation(input_tensor, perm, output_shape, num_dimensions,
                       /*output_axis=*/0, &input_indices, &new_values);
    auto result_type =
        RankedTensorType::get(output_shape, output_type.getElementType());
    auto values_type = RankedTensorType::get(
        output_shape, output_type.getElementType()
                          .cast<quant::UniformQuantizedType>()
                          .getStorageType());
    rewriter.replaceOpWithNewOp<QConstOp>(
        op, TypeAttr::get(result_type),
        DenseIntElementsAttr::get(values_type, new_values));
    return success();
  }
};

// Fold constant quantized Reshape ops.
struct FoldReshapeOp : public OpRewritePattern<ReshapeOp> {
  // Does not take ownership of context, which must refer to a valid value that
  // outlives this object.
  explicit FoldReshapeOp(MLIRContext* context)
      : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(ReshapeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def_op = op.getInput().getDefiningOp();
    auto qconst_op = llvm::dyn_cast_or_null<QConstOp>(def_op);
    if (qconst_op == nullptr) return failure();

    auto dense_elements =
        qconst_op.getValue().dyn_cast_or_null<DenseElementsAttr>();
    if (dense_elements == nullptr) return failure();

    // Handle per tensor cases only.
    if (!(getElementTypeOrSelf(op.getType()))
             .isa<quant::UniformQuantizedType>()) {
      return failure();
    }

    // Remove identity reshape with both static result and input shape.
    auto result_type = op.getType().cast<ShapedType>();
    auto input_type = op.getInput().getType().cast<ShapedType>();

    // Constant folding
    // If the result type isn't static, tries to derive the result type from
    // the #2 operand.
    if (!result_type.hasStaticShape()) {
      DenseIntElementsAttr shape_elements;
      if (!matchPattern(op.getShape(), m_Constant(&shape_elements)))
        return failure();

      SmallVector<int64_t, 4> shape_data;
      for (const APInt& it : shape_elements.getValues<APInt>()) {
        shape_data.push_back(it.getSExtValue());
      }
      result_type =
          RankedTensorType::get(shape_data, input_type.getElementType());
    }
    auto values_type = RankedTensorType::get(
        result_type.getShape(), result_type.getElementType()
                                    .cast<quant::UniformQuantizedType>()
                                    .getStorageType());

    DenseElementsAttr reshaped_elements = dense_elements.reshape(values_type);
    rewriter.replaceOpWithNewOp<QConstOp>(op, TypeAttr::get(result_type),
                                          reshaped_elements);
    return success();
  }
};

// Removes operations with side effect (i.e. LSTM, SVDF) that have dangling
// output.
template <typename OpTy>
struct PruneUnusedOpsWithSideEffect : public OpRewritePattern<OpTy> {
 public:
  explicit PruneUnusedOpsWithSideEffect(
      MLIRContext* context, const quant::CustomOpMap& custom_op_map = {})
      : OpRewritePattern<OpTy>(context), custom_op_map(custom_op_map) {}

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
    // Remove if the custom op is in the provided map and is NoSideEffect.
    auto custom_op = llvm::isa<CustomOp>(op);
    if (custom_op) {
      auto q = llvm::cast<CustomOp>(op);
      std::string op_name = q.getCustomCode().str();
      if ((custom_op_map.find(op_name) == custom_op_map.end()) ||
          !custom_op_map.find(op_name)->second.no_side_effect)
        return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
  quant::CustomOpMap custom_op_map;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_post_quantize.inc"

void PostQuantizePass::runOnOperation() {
  if (!enable_custom_op_no_side_effect_.empty()) {
    ParseCustomOpSpecs(enable_custom_op_no_side_effect_,
                       quant::CustomOpUpdateOptions::kNoSideEffect,
                       custom_op_map_);
  }

  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.add<quant::FoldTrivalRequantizeOp<QuantizeOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::LSTMOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::UnidirectionalSequenceLSTMOp>>(
      ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::SVDFOp>>(ctx);
  patterns.add<PruneUnusedOpsWithSideEffect<TFL::CustomOp>>(ctx,
                                                            custom_op_map_);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  if (!emit_quant_adaptor_ops_) {
    RemoveQuantizationAdaptorOps(getOperation());
  }

  RewritePatternSet phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.add<quant::FoldTrivalRequantizeOp<QuantizeOp>,
                       RemoveVolatileOps<kPreserveInputsAndOutputs>,
                       FoldTransposeOp, FoldReshapeOp>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_2_patterns));
}

void PostQuantizeRemoveQDQPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  auto* ctx = func.getContext();
  TFL::populateWithGenerated(patterns);
  patterns.add<RemoveVolatileOps<kPreserveNone>>(ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PostQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass(
    bool emit_quant_adaptor_ops, const quant::CustomOpMap& custom_op_map) {
  return std::make_unique<PostQuantizePass>(emit_quant_adaptor_ops,
                                            custom_op_map);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizePass() {
  return std::make_unique<PostQuantizePass>();
}

// Creates an instance of the TensorFlow Lite dialect PostQuantizeRemoveQDQ
// pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePostQuantizeRemoveQDQPass() {
  return std::make_unique<PostQuantizeRemoveQDQPass>();
}

}  // namespace TFL
}  // namespace mlir
