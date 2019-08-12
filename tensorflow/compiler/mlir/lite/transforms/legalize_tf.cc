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

// This transformation pass converts operations in TensorFlow dialect into
// operations that are legal in the TensorFlow Lite dialect.  Operations that
// can be legalized to TensorFlow Lite dialect with simple replacements are part
// of this pass and other operations that may create extra ops should be part of
// the PrepareTF pass which should be run before this pass.  That way any
// constant folding opportunities from the extra ops can be exploited by the
// constant folding support for the TensorFlow ops.

#include <climits>

#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/UniformSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual LegalizeTF Pass.
namespace {

// Legalize operations in functions.
struct LegalizeTF : public FunctionPass<LegalizeTF> {
  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_legalize_tf.inc"

#define DECL_CONVERT_OP(tf_op)                                             \
  struct ConvertTF##tf_op##Op : public RewritePattern {                    \
    explicit ConvertTF##tf_op##Op(MLIRContext* context)                    \
        : RewritePattern(TF::tf_op##Op::getOperationName(), 1, context) {} \
    PatternMatchResult matchAndRewrite(                                    \
        Operation* op, PatternRewriter& rewriter) const override;          \
  }

// TODO(antiagainst): Define this pattern in a table-driven manner once variadic
// operands are properly supported in declarative rewrite rule specification.

DECL_CONVERT_OP(Concat);
DECL_CONVERT_OP(ConcatV2);
DECL_CONVERT_OP(MatMul);
DECL_CONVERT_OP(Pack);
DECL_CONVERT_OP(Split);
DECL_CONVERT_OP(SplitV);
DECL_CONVERT_OP(Unpack);

#undef DECL_CONVERT_OP

PatternMatchResult ConvertTFConcatOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concat_op = cast<TF::ConcatOp>(op);

  SmallVector<Value*, 4> values(tf_concat_op.values());
  auto output_type = tf_concat_op.output()->getType();
  // Extract axis attribute from constant concat_dims tensor
  ElementsAttr axis;
  if (!matchPattern(tf_concat_op.concat_dim(), m_Constant(&axis)))
    return matchFailure();

  StringAttr fused_activation_function =
      StringAttr::get("NONE", rewriter.getContext());
  rewriter.replaceOpWithNewOp<TFL::ConcatenationOp>(
      op, output_type, values, mlir::TFL::ExtractSingleElementAsInteger(axis),
      fused_activation_function);
  return matchSuccess();
}

PatternMatchResult ConvertTFConcatV2Op::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_concat_op = cast<TF::ConcatV2Op>(op);

  SmallVector<Value*, 4> values(tf_concat_op.values());
  auto output_type = tf_concat_op.output()->getType();
  // Extract axis attribute from constant axis tensor
  ElementsAttr axis;
  if (!matchPattern(tf_concat_op.axis(), m_Constant(&axis)))
    return matchFailure();

  StringAttr fused_activation_function =
      StringAttr::get("NONE", rewriter.getContext());
  rewriter.replaceOpWithNewOp<ConcatenationOp>(
      op, output_type, values, ExtractSingleElementAsInteger(axis),
      fused_activation_function);
  return matchSuccess();
}

// The following is effectively:
// def : Pat<
//   (TF_MatMulOp $a, $b, ConstBoolAttrFalse:$transpose_a,
//      ConstBoolAttrTrue:$transpose_b),
//   (TFL_FullyConnectedOp:$__0 $a, $b,
//     NoInput.pattern, TFL_AF_None, TFL_FCWO_Default, ConstBoolAttrFalse)>;
PatternMatchResult ConvertTFMatMulOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_matmul_op = cast<TF::MatMulOp>(op);
  if (tf_matmul_op.transpose_a()) return matchFailure();
  if (!tf_matmul_op.transpose_b()) return matchFailure();

  Type output_type = tf_matmul_op.getResult()->getType();
  // TODO(jpienaar): Follow up post shuffle discussion.
  auto no_input = rewriter.create<ConstantOp>(
      op->getLoc(), rewriter.getNoneType(), rewriter.getUnitAttr());
  auto fc_op = rewriter.create<FullyConnectedOp>(
      op->getLoc(), ArrayRef<Type>{output_type}, op->getOperand(0),
      op->getOperand(1), no_input, rewriter.getStringAttr("NONE"),
      rewriter.getStringAttr("DEFAULT"), rewriter.getBoolAttr(false));
  rewriter.replaceOp(op, {fc_op.getResult(0)});
  return matchSuccess();
}

PatternMatchResult ConvertTFPackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_pack_op = cast<TF::PackOp>(op);

  SmallVector<Value*, 4> values(tf_pack_op.values());
  auto output_type = tf_pack_op.output()->getType();
  auto values_count = rewriter.getI32IntegerAttr(tf_pack_op.N().getZExtValue());
  // Axis can be negative.
  auto axis = rewriter.getI32IntegerAttr(tf_pack_op.axis().getSExtValue());

  rewriter.replaceOpWithNewOp<PackOp>(op, output_type, values, values_count,
                                      axis);
  return matchSuccess();
}

PatternMatchResult ConvertTFSplitOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_split_op = cast<TF::SplitOp>(op);

  auto output_types = functional::map([](Value* v) { return v->getType(); },
                                      tf_split_op.output());
  // Number of splits cannot be negative.
  auto num_split =
      rewriter.getI32IntegerAttr(tf_split_op.num_split().getZExtValue());

  rewriter.replaceOpWithNewOp<TFL::SplitOp>(op, output_types,
                                            tf_split_op.split_dim(),
                                            tf_split_op.value(), num_split);
  return matchSuccess();
}

PatternMatchResult ConvertTFSplitVOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_splitv_op = cast<TF::SplitVOp>(op);

  auto output_types = functional::map([](Value* v) { return v->getType(); },
                                      tf_splitv_op.output());
  // Number of splits cannot be negative.
  auto num_split =
      rewriter.getI32IntegerAttr(tf_splitv_op.num_split().getZExtValue());

  rewriter.replaceOpWithNewOp<TFL::SplitVOp>(
      op, output_types, tf_splitv_op.value(), tf_splitv_op.size_splits(),
      tf_splitv_op.split_dim(), num_split);
  return matchSuccess();
}

PatternMatchResult ConvertTFUnpackOp::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  auto tf_unpack_op = cast<TF::UnpackOp>(op);

  auto* input = tf_unpack_op.value();
  auto output_types = functional::map([](Value* v) { return v->getType(); },
                                      tf_unpack_op.output());
  auto num = rewriter.getI32IntegerAttr(tf_unpack_op.num().getZExtValue());
  // Axis can be negative.
  auto axis = rewriter.getI32IntegerAttr(tf_unpack_op.axis().getSExtValue());

  rewriter.replaceOpWithNewOp<UnpackOp>(op, output_types, input, num, axis);
  return matchSuccess();
}

void LegalizeTF::runOnFunction() {
  OwningRewritePatternList patterns;
  auto* ctx = &getContext();
  auto func = getFunction();

  // Add the generated patterns to the list.
  populateWithGenerated(ctx, &patterns);
  patterns.insert<ConvertTFConcatOp, ConvertTFConcatV2Op, ConvertTFMatMulOp,
                  ConvertTFPackOp, ConvertTFSplitOp, ConvertTFSplitVOp,
                  ConvertTFUnpackOp>(ctx);
  applyPatternsGreedily(func, patterns);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect LegalizeTF pass.
FunctionPassBase* CreateLegalizeTFPass() { return new LegalizeTF(); }

static PassRegistration<LegalizeTF> pass(
    "tfl-legalize-tf", "Legalize from TensorFlow to TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
