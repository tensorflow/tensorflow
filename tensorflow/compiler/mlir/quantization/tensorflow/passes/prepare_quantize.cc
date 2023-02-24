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
// Copied and modified from
// //third_party/tensorflow/compiler/mlir/lite/transforms/prepare_quantize.cc
// This transformation pass applies quantization propagation on TF dialect.
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace quant {

namespace {

using QuantMethod =
    tensorflow::quantization::QuantizationMethod::ExperimentalMethod;

// Applies prepare quantization on the model in TF dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass
    : public PassWrapper<PrepareQuantizePass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<TF::TensorFlowDialect, ::mlir::quant::QuantizationDialect,
                    ::mlir::quantfork::QuantizationForkDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizePass)

  // Constructor used by the PassRegistration and enforce uint8 quantization.
  // This is only used by test.
  explicit PrepareQuantizePass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizePass(const QuantizationSpecs& quant_specs,
                               QuantMethod quantization_method)
      : quant_specs_(quant_specs) {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
    enable_per_channel_quantization_ = !quant_specs_.disable_per_channel;
    enable_post_training_quantize_ =
        (quantization_method ==
         tensorflow::quantization::QuantizationMethod::STATIC_RANGE);
  }

  PrepareQuantizePass(const PrepareQuantizePass& other) {
    quant_specs_ = other.quant_specs_;
    enable_post_training_quantize_ = other.enable_post_training_quantize_;
    enable_per_channel_quantization_ = !quant_specs_.disable_per_channel;
  }

  explicit PrepareQuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {
    enable_post_training_quantize_ = quant_specs.post_training_quantization;
  }

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-prepare-quantize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Prepare TF dialect for quantization";
  }

  void runOnOperation() override;

 private:
  // Set the quantization parameters of the input nodes. These parameters are
  // converted from the user specified input value ranges. The input nodes with
  // non-float tensor types will be skipped because they are not quantizable.
  // Return true if number of input nodes doesn't equal to that of the input
  // ranges.
  bool SetInputNodesQuantizationParams(func::FuncOp func);

  // The function might contain more stats ops than required, and it will
  // introduce requantize if the calibration stats have conflicts. This method
  // tries to remove all the redundant stats ops.
  bool RemoveRedundantStats(func::FuncOp func);

  // Verify the quantization specification is expected for quantizing the
  // current function.
  bool IsLegalQuantSpecs(func::FuncOp func) {
    if (func.getName() == quant_specs_.target_func) {
      return func.getNumArguments() == quant_specs_.input_ranges.size();
    }
    return true;
  }

  // Get the min and max values from the quantization specification for the
  // current function and argument index. Uses default values if the function
  // is specified in the `quantize_allowlist`.
  std::pair<llvm::Optional<double>, llvm::Optional<double>>
  GetMinMaxValuesForArgument(llvm::StringRef func_name, int index) {
    if (func_name == quant_specs_.target_func) {
      return quant_specs_.input_ranges[index];
    } else {
      return {0.0, 255.0};
    }
  }

  // Apply some sanity check and report some warnings for those who don't follow
  // the best quantization practice. This also fixes some simple violations.
  void SanityCheckAndAdjustment(func::FuncOp func);

  // Whether the func contains Quantize ops. This is used to determine whether
  // to use the quantization parameters from the fixed output range property.
  bool ContainsQuantizeOps(func::FuncOp func);

  QuantizationSpecs quant_specs_;

  Option<bool> enable_post_training_quantize_{
      *this, "post-training-quantize", llvm::cl::init(false),
      llvm::cl::desc("Enable post training quantization. Only used in tests.")};

  // A local flag is needed for testing conditions in
  // prepare_quantize_ptq_per_channel.mlir.
  Option<bool> enable_per_channel_quantization_{
      *this, "enable-per-channel-quantization", llvm::cl::init(false),
      llvm::cl::desc("Whether enable per-channel quantized weights.")};
};

bool PrepareQuantizePass::SetInputNodesQuantizationParams(func::FuncOp func) {
  StringRef func_name = func.getName();
  auto has_quantize_op = [&](const Value arg) {
    return (arg.hasOneUse() &&
            llvm::isa<quantfork::QuantizeCastOp>(*arg.user_begin()));
  };

  bool need_to_set_input_nodes_quantization_params = false;
  for (const BlockArgument arg : func.getArguments()) {
    auto shaped = arg.getType().dyn_cast<ShapedType>();
    if (shaped && shaped.getElementType().isa<FloatType>() &&
        !has_quantize_op(arg)) {
      need_to_set_input_nodes_quantization_params = true;
      break;
    }
  }

  if (!need_to_set_input_nodes_quantization_params) {
    return false;
  }

  // If the validation fails, the pass should stop immediately.
  if (!IsLegalQuantSpecs(func)) {
    return true;
  }

  OpBuilder builder(func);
  bool is_signed = quant_specs_.IsSignedInferenceType();
  IntegerAttr num_bits =
      builder.getI32IntegerAttr(quant_specs_.GetQuantizationTypeWidth());
  BoolAttr narrow_range = builder.getBoolAttr(false);

  auto add_quantize_op = [&](Location loc, Type input_type, Block* block,
                             Block::iterator insertion_point, Value arg,
                             int i) {
    if (auto shaped = input_type.dyn_cast<ShapedType>()) {
      if (shaped.getElementType().isa<FloatType>()) {
        // If there are existing quantize ops, they are from training and we
        // should respect them.
        if (has_quantize_op(arg)) {
          return;
        }

        auto min_max = GetMinMaxValuesForArgument(func_name, i);
        // The input min/max or mean/std are not specified, then skip.
        if (!min_max.first.has_value() || !min_max.second.has_value()) return;

        TypeAttr params = quant::GetQuantizedTypeAttr(
            builder, input_type, builder.getF64FloatAttr(min_max.first.value()),
            builder.getF64FloatAttr(min_max.second.value()),
            /*quant_dim=*/-1, num_bits, narrow_range, is_signed);
        builder.setInsertionPoint(block, insertion_point);
        auto q_op = builder.create<quantfork::QuantizeCastOp>(
            loc, params.getValue(), arg);
        auto dq_op = builder.create<quantfork::DequantizeCastOp>(
            loc, input_type, q_op.getResult());
        arg.replaceAllUsesWith(dq_op.getResult());
        q_op.setOperand(arg);
      }
    }
  };

  for (int i = 0, e = func.getNumArguments(); i != e; ++i) {
    BlockArgument arg = func.getArgument(i);
    auto* arg_block = arg.getOwner();
    add_quantize_op(arg.getLoc(), arg.getType(), arg_block,
                    std::next(arg_block->begin(), i), arg, i);
  }

  return false;
}

bool PrepareQuantizePass::RemoveRedundantStats(func::FuncOp func) {
  return RemoveRedundantStatsOps(func, GetTFOpQuantSpec, GetTfQuantScaleSpec);
}

static Value Quantized(Operation* user) {
  if (auto q = llvm::dyn_cast_or_null<quantfork::QuantizeCastOp>(user)) {
    if (auto dq = llvm::dyn_cast_or_null<quantfork::DequantizeCastOp>(
            *q.getResult().user_begin())) {
      return dq.getResult();
    }
  }
  return {};
}

void PrepareQuantizePass::SanityCheckAndAdjustment(func::FuncOp func) {
  // If an op output has two users: one of them is a quantize op and another
  // one is returned directly, we decide to return the quantized result instead,
  // so this op can be quantized. This is only applied on the returned result
  // because the error will not be accumulated.

  func.walk([&](func::ReturnOp ret) {
    int i = 0;
    for (Value returned : ret.getOperands()) {
      llvm::SmallVector<Value, 4> quantized;
      for (auto user : returned.getUsers()) {
        if (auto q = Quantized(user)) {
          quantized.push_back(q);
        }
      }
      if (quantized.size() == 1) {
        ret.setOperand(i, quantized.front());
      }
      i++;
    }
  });

  // Check for  (Quant (Dequant $in), $qA) "qdq" pairs that couldn't be
  // eliminated at this point.  This only occurs for the pattern
  //      (Quant (Dequant (Quant $in, $qB)), $qA)   $qB != $qA
  // where the  qdq pair denotes a non-trivial requantization of an
  // already quantized value. Since this makes little sense (directly quantizing
  // (Quant $in, $qA) would introduce less quantization noise) the likely cause
  // is an minor error in constructing the original network model that
  // introduced back-to-back Fake Quantization operations. Hence: emit a
  // warning. N.b. at this point we're (teporarility) in the quantization
  // dialect (presumably enable re-use in xla etc) quantfork::*QuantizeCastOp
  // we're matching here.
  //
  func.walk([&](quantfork::QuantizeCastOp q_op) {
    // If up with end up with
    auto dq_op = dyn_cast_or_null<quantfork::DequantizeCastOp>(
        q_op.getOperand().getDefiningOp());
    if (!dq_op) {
      return;
    }
    auto dq_arg = dq_op.getOperand();

    if (!dq_arg.hasOneUse()) {
      // The initial quantization is used someplace else ... so it might be
      // reasonable for it to requantized for another purpose.
      // Ideally would want to still check whether requantization narrows
      // rather than widens the representation.
      return;
    }

    // Invariant:
    // isa<quantfork::QuantizeCastOp>(dq_arg.getDefiningOp()) -->
    // getdq_arg.getType() != q_op.getResult().getType()
    //
    // as otherwise qdq pair would have been optimized away.
    auto qd_arg_def_q_op =
        dyn_cast_or_null<quantfork::QuantizeCastOp>(dq_arg.getDefiningOp());
    if (!qd_arg_def_q_op) {
      return;
    }

    qd_arg_def_q_op.emitWarning()
        << " quantizer's output has another quantizer (" << q_op.getLoc()
        << ") as consumer - intentional?";
  });
}

// Merges consecutive QuantizeCast ops. For example, the following case:
// %1 = tf.QuantizeCastOp(%0) : f32 -> qtype1
// %2 = tf.QuantizeCastOp(%1) : qtype1 -> qtype2
// %3 = tf.QuantizedOp1(%1)
// %4 = tf.QuantizedOp2(%2)
// will be tranformed to:
// %1 = tf.QuantizeCastOp(%0) : f32 -> qtype1
// %2 = tf.QuantizeCastOp(%0) : f32 -> qtype2
// %3 = tf.QuantizedOp1(%1)
// %4 = tf.QuantizedOp2(%2)
// Converting from f32 -> qtype1 -> qtype2 will add unexpected quantization
// lost for %2. This pattern avoids that by converting from f32 -> qtype2
// directly.
class MergeConsecutiveQuantizeCast
    : public mlir::OpRewritePattern<quantfork::QuantizeCastOp> {
 public:
  explicit MergeConsecutiveQuantizeCast(MLIRContext* context)
      : OpRewritePattern<quantfork::QuantizeCastOp>(context) {}

 private:
  LogicalResult matchAndRewrite(quantfork::QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
    auto preceding_qcast =
        q_op.getArg().getDefiningOp<quantfork::QuantizeCastOp>();
    if (!preceding_qcast) return failure();

    auto new_qcast = rewriter.create<quantfork::QuantizeCastOp>(
        q_op.getLoc(), q_op.getType(), preceding_qcast.getArg());
    new_qcast->setAttr(kVolatileOpAttrName, rewriter.getUnitAttr());
    q_op->replaceAllUsesWith(new_qcast);
    return success();
  }
};

bool PrepareQuantizePass::ContainsQuantizeOps(func::FuncOp func) {
  for (const auto& op : func.getOps()) {
    if (llvm::isa<quantfork::DequantizeCastOp>(op)) return true;
  }
  return false;
}

using PrepareQuantStats =
    quant::ConvertStatsToQDQs<quantfork::QuantizeCastOp,
                              quantfork::DequantizeCastOp>;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_quantize.inc"

void PrepareQuantizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  quant_specs_.post_training_quantization = enable_post_training_quantize_;
  if (quant_specs_.post_training_quantization) {
    RemoveRedundantStats(func);
  } else {
    // Set the quantization parameters for the quantizable input nodes. If this
    // failed, return the function immediately. This is only required for
    // quantization aware training model conversion.
    if (SetInputNodesQuantizationParams(func)) {
      return;
    }
  }

  bool is_signed = quant_specs_.IsSignedInferenceType();
  int bit_width = quant_specs_.GetQuantizationTypeWidth();
  // When this is true, the quantizer will try its best to extract the
  // quantization parameters from the op quantization property and constant
  // content. This is also set to true when the `quantize_allowlist` and
  // `quantize_signed` test flags are enabled.
  bool eager_quantize = ContainsQuantizeOps(func);
  // Infer the tensor range for the activation ops and weight constants unless
  // it is disabled explicitly.
  bool infer_tensor_range =
      (quant_specs_.post_training_quantization || eager_quantize) &&
      !quant_specs_.disable_infer_tensor_range;

  // During the legalization, unsigned quantized type is used, so we have to
  // convert all of them to signed.
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  patterns.add<quant::ConvertUnsignedToSigned<quantfork::QuantizeCastOp>>(ctx);
  // Convert quant stats to int8 quantization parameters.
  // Currently, only activation stats are imported, so narrow_range = false.
  patterns.add<PrepareQuantStats>(bit_width, false, true,
                                  /*legacy_float_scale=*/false, ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }

  SanityCheckAndAdjustment(func);

  // Finally, the quantization parameters can be propagated to the rest of the
  // values (tensors).
  ApplyQuantizationParamsPropagation(
      func, is_signed, /*bit_width=*/8, !enable_per_channel_quantization_,
      GetTFOpQuantSpec, GetTfQuantScaleSpec, infer_tensor_range,
      quant_specs_.legacy_float_scale);

  RewritePatternSet patterns2(ctx);
  patterns2.add<MergeConsecutiveQuantizeCast>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns2)))) {
    signalPassFailure();
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs, QuantMethod quantization_method) {
  return std::make_unique<PrepareQuantizePass>(quant_specs,
                                               quantization_method);
}

static PassRegistration<PrepareQuantizePass> pass;

}  // namespace quant
}  // namespace mlir
