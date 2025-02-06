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

// This transformation pass applies quantization propagation on TFLite dialect.
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/Quant.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_config.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_driver.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {
#define GEN_PASS_DEF_PREPAREQUANTIZEPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

auto* tflite_quantizer_usage_stats = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/lite/quantization/transforms/stats",
    "The number of quantization pass invocations.", "path");

// Applies prepare quantization on the model in TFL dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass
    : public impl::PrepareQuantizePassBase<PrepareQuantizePass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizePass)

  // Constructor used by the PassRegistration and enforce uint8 quantization.
  // This is only used by test.
  explicit PrepareQuantizePass() : use_quantization_flags_(true) {}

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizePass(const quant::QuantizationSpecs& quant_specs)
      : use_quantization_flags_(false), quant_specs_(quant_specs) {}

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
      return (quant_specs_.disable_set_input_nodes_quantization_params ||
              func.getNumArguments() == quant_specs_.input_ranges.size());
    }
    return true;
  }

  // Get the min and max values from the quantization specification for the
  // current function and argument index. Uses default values if the function
  // is specified in the `quantize_allowlist`.
  std::pair<std::optional<double>, std::optional<double>>
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

  bool use_quantization_flags_;
  quant::QuantizationSpecs quant_specs_;
};

bool PrepareQuantizePass::SetInputNodesQuantizationParams(func::FuncOp func) {
  if (quant_specs_.disable_set_input_nodes_quantization_params) {
    return false;
  }

  StringRef func_name = func.getName();
  auto& target_func = quant_specs_.target_func;
  // Skip this function because it isn't the target function from the spec or
  // in the function while list.
  if (target_func != func_name &&
      !llvm::is_contained(quantize_allowlist_, func_name)) {
    return false;
  }
  auto has_quantize_op = [&](const Value arg) {
    return (arg.hasOneUse() &&
            llvm::isa<quantfork::QuantizeCastOp>(*arg.user_begin()));
  };

  bool need_to_set_input_nodes_quantization_params = false;
  for (const BlockArgument arg : func.getArguments()) {
    auto shaped = mlir::dyn_cast<ShapedType>(arg.getType());
    if (shaped && mlir::isa<FloatType>(shaped.getElementType()) &&
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
    if (auto shaped = mlir::dyn_cast<ShapedType>(input_type)) {
      if (mlir::isa<FloatType>(shaped.getElementType())) {
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

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

bool PrepareQuantizePass::RemoveRedundantStats(func::FuncOp func) {
  return RemoveRedundantStatsOps(
      func, std::bind(GetOpQuantSpec, std::placeholders::_1,
                      quant_specs_.disable_per_channel_for_dense_layers));
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

  // We prefer to placing quantization emulation ops on the results of the
  // concat ops.
  func.walk([&](ConcatenationOp concat) {
    if (concat.getOutput().hasOneUse() &&
        Quantized(*concat.getOutput().user_begin())) {
      return;
    }
    concat.emitWarning(
        "Missing quantization parameter on the output might introduce "
        "quantization error!");
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

bool PrepareQuantizePass::ContainsQuantizeOps(func::FuncOp func) {
  for (const auto& op : func.getOps()) {
    if (llvm::isa<quantfork::DequantizeCastOp>(op)) return true;
  }
  return false;
}

using PrepareQuantStats =
    quant::ConvertStatsToQDQs<quantfork::QuantizeCastOp,
                              quantfork::DequantizeCastOp>;

void PrepareQuantizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();
  ScopedTFLQuantOpsToMlirQuantOpsConverter converter(func);
  if (use_quantization_flags_) {
    quant_specs_.inference_type = GetQuantizedInferenceType(
        this->quantize_signed_, this->activation_number_of_bits_);
    if (quant_specs_.inference_type == tensorflow::DT_INVALID) {
      func.emitError() << "prepare-quantize pass failed: unsupported "
                          "inference type specification";
      signalPassFailure();
      return;
    }
    quant_specs_.post_training_quantization = post_training_quantize_;
    quant_specs_.legacy_float_scale = legacy_float_scale_;
    quant_specs_.disable_set_input_nodes_quantization_params =
        disable_set_input_nodes_quantization_params_;
    for (const auto& ir : input_ranges_) {
      std::pair<std::string, std::string> input_range = absl::StrSplit(ir, '|');
      std::optional<double> optional_min;
      std::optional<double> optional_max;

      if (!input_range.first.empty()) {
        double min;
        (void)absl::SimpleAtod(input_range.first, &min);
        optional_min = min;
      }

      if (!input_range.second.empty()) {
        double max;
        (void)absl::SimpleAtod(input_range.second, &max);
        optional_max = max;
      }

      quant_specs_.input_ranges.emplace_back(optional_min, optional_max);
    }
  }

  if (quant_specs_.post_training_quantization) {
    tflite_quantizer_usage_stats->GetCell("post_training")->IncrementBy(1);
    RemoveRedundantStats(func);
  } else {
    tflite_quantizer_usage_stats->GetCell("during_training")->IncrementBy(1);
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
  bool eager_quantize = ContainsQuantizeOps(func) ||
                        (!quantize_allowlist_.empty() || quantize_signed_);
  // Infer the tensor range for the activation ops and weight constants unless
  // it is disabled explicitly.
  bool infer_tensor_range =
      (quant_specs_.post_training_quantization || eager_quantize) &&
      !quant_specs_.disable_infer_tensor_range;

  // LSTM's restrict_scale requirement should be handled before converting stats
  // to Q-DQ ops. The pattern is applied for non-PTQ case to make op ordering
  // consistent. Otherwise some FileCheck tests would fail.
  RewritePatternSet patterns_1(&getContext());
  if (quant_specs_.post_training_quantization) {
    patterns_1.add<PrepareLstmOutputScale<LSTMOp>>(ctx);
    patterns_1.add<PrepareLstmOutputScale<UnidirectionalSequenceLSTMOp>>(ctx);
  }
  if (is_qdq_conversion_ ||
      quant_specs_.qdq_conversion_mode != quant::QDQConversionMode::kQDQNone) {
    patterns_1.add<PropagateTransposedPerAxisQuantDim>(ctx);
  }
  (void)applyPatternsGreedily(func, std::move(patterns_1));

  // During the legalization, unsigned quantized type is used, so we have to
  // convert all of them to signed.
  RewritePatternSet patterns_2(&getContext());
  if (is_signed) {
    patterns_2.add<quant::ConvertUnsignedToSigned<quantfork::QuantizeCastOp>>(
        ctx);
  }
  // Convert quant stats to int8, unit8, int16 quantization parameters.
  // Currently, only activation stats are imported, so narrow_range = false.
  // TODO(b/266524882): Support narrow_range in TFLite converter(ODML
  // converter).
  patterns_2.add<PrepareQuantStats>(bit_width, /*narrow_range=*/false,
                                    is_signed, quant_specs_.legacy_float_scale,
                                    ctx);

  if (quant_specs_.post_training_quantization) {
    patterns_2.add<ConvertLstmStatsToQDQs<LSTMOp>>(ctx, quant_specs_);
    patterns_2.add<ConvertLstmStatsToQDQs<UnidirectionalSequenceLSTMOp>>(
        ctx, quant_specs_);
    patterns_2.add<ConvertSvdfStatsToQDQs>(ctx, quant_specs_);
  }
  (void)applyPatternsGreedily(func, std::move(patterns_2));

  SanityCheckAndAdjustment(func);

  // Bind the getter with the fixed configuration parameter for the correct
  // quantization settings of the ops.
  std::function<std::unique_ptr<quant::OpQuantSpec>(Operation*)>
      op_quant_spec_getter =
          std::bind(GetOpQuantSpec, std::placeholders::_1,
                    quant_specs_.disable_per_channel_for_dense_layers);

  // Finally, the quantization parameters can be propagated to the rest of the
  // values (tensors).
  ApplyQuantizationParamsPropagation(
      func, is_signed, bit_width,
      disable_per_channel_ || quant_specs_.disable_per_channel,
      op_quant_spec_getter, infer_tensor_range, quant_specs_.legacy_float_scale,
      (is_qdq_conversion_ ||
       quant_specs_.qdq_conversion_mode != quant::QDQConversionMode::kQDQNone));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    const quant::QuantizationSpecs& quant_specs) {
  return std::make_unique<PrepareQuantizePass>(quant_specs);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass() {
  return std::make_unique<PrepareQuantizePass>();
}

}  // namespace TFL
}  // namespace mlir
