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
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/util.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/quant_spec.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// NOLINTNEXTLINE
static llvm::cl::opt<bool> post_training_quantize_flag(
    "quant-test-post-training-quantize", llvm::cl::value_desc("bool"),
    llvm::cl::desc("enable post training quantization. Only used in tests"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> disable_per_channel(
    "quant-disable-per-channel", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether disable per-channel quantized weights."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace quant {

namespace {

// Applies prepare quantization on the model in TF dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass
    : public PassWrapper<PrepareQuantizePass, OperationPass<func::FuncOp>> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry
        .insert<TF::TensorFlowDialect, ::mlir::quant::QuantizationDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareQuantizePass)

  // Constructor used by the PassRegistration and enforce uint8 quantization.
  // This is only used by test.
  explicit PrepareQuantizePass() {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
    quant_specs_.post_training_quantization = post_training_quantize_flag;
  }

  explicit PrepareQuantizePass(QuantizationMethod quantization_method) {
    quant_specs_.inference_type = tensorflow::DT_QINT8;
    quant_specs_.post_training_quantization =
        (quantization_method == QuantizationMethod::kPostTrainingQuantization);
  }

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {}

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
};

bool PrepareQuantizePass::SetInputNodesQuantizationParams(func::FuncOp func) {
  StringRef func_name = func.getName();
  auto has_quantize_op = [&](const Value arg) {
    return (arg.hasOneUse() &&
            llvm::isa<quant::QuantizeCastOp>(*arg.user_begin()));
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
        if (!min_max.first.hasValue() || !min_max.second.hasValue()) return;

        TypeAttr params = quant::GetQuantizedTypeAttr(
            builder, input_type,
            builder.getF64FloatAttr(min_max.first.getValue()),
            builder.getF64FloatAttr(min_max.second.getValue()),
            /*quant_dim=*/-1, num_bits, narrow_range, is_signed);
        builder.setInsertionPoint(block, insertion_point);
        auto q_op =
            builder.create<quant::QuantizeCastOp>(loc, params.getValue(), arg);
        auto dq_op = builder.create<quant::DequantizeCastOp>(loc, input_type,
                                                             q_op.getResult());
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

// TODO(b/213253905): set appropriate quant spec getter
std::unique_ptr<OpQuantSpec> GetOpQuantSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantSpec>();
  if (auto call_op = dyn_cast<TF::PartitionedCallOp>(op)) {
    StringRef function_name =
        call_op.fAttr().cast<FlatSymbolRefAttr>().getValue();
    if (!function_name.startswith("composite_")) {
      return spec;
    }
    if (function_name.contains("depthwise_conv2d_with_bias")) {
      spec->biases_params[2] = {{0, 1}, quant::GetUniformQuantizedTypeForBias};
      spec->coeff_op_quant_dim[0] = 2;
    } else if (function_name.contains("conv2d_with_bias")) {
      spec->biases_params[2] = {{0, 1}, quant::GetUniformQuantizedTypeForBias};
      spec->coeff_op_quant_dim[0] = 3;
    } else if (function_name.contains("matmul_with_bias")) {
      spec->biases_params[2] = {{0, 1}, quant::GetUniformQuantizedTypeForBias};
      spec->coeff_op_quant_dim[0] = -1;
    }
  }
  return spec;
}

bool PrepareQuantizePass::RemoveRedundantStats(func::FuncOp func) {
  return RemoveRedundantStatsOps(func, GetOpQuantSpec, GetTfQuantScaleSpec);
}

static Value Quantized(Operation* user) {
  if (auto q = llvm::dyn_cast_or_null<quant::QuantizeCastOp>(user)) {
    if (auto dq = llvm::dyn_cast_or_null<quant::DequantizeCastOp>(
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

  func.walk([&](ReturnOp ret) {
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
  // dialect (presumably enable re-use in xla etc) quant::*QuantizeCastOp
  // we're matching here.
  //
  func.walk([&](quant::QuantizeCastOp q_op) {
    // If up with end up with
    auto dq_op = dyn_cast_or_null<quant::DequantizeCastOp>(
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
    // isa<quant::QuantizeCastOp>(dq_arg.getDefiningOp()) -->
    // getdq_arg.getType() != q_op.getResult().getType()
    //
    // as otherwise qdq pair would have been optimized away.
    auto qd_arg_def_q_op =
        dyn_cast_or_null<quant::QuantizeCastOp>(dq_arg.getDefiningOp());
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
    if (llvm::isa<quant::DequantizeCastOp>(op)) return true;
  }
  return false;
}

using PrepareQuantStats =
    quant::ConvertStatsToQDQs<quant::QuantizeCastOp, quant::DequantizeCastOp>;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_quantize.inc"

void PrepareQuantizePass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

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
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.add<quant::ConvertUnsignedToSigned<quant::QuantizeCastOp>>(ctx);
  // Convert quant stats to int8 quantization parameters.
  // Currently, only activation stats are imported, so narrow_range = false.
  patterns.add<PrepareQuantStats>(bit_width, false, true,
                                  /*legacy_float_scale=*/false, ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  SanityCheckAndAdjustment(func);

  // Finally, the quantization parameters can be propagated to the rest of the
  // values (tensors).
  ApplyQuantizationParamsPropagation(
      func, is_signed, disable_per_channel || quant_specs_.disable_per_channel,
      GetOpQuantSpec, GetTfQuantScaleSpec, infer_tensor_range,
      quant_specs_.legacy_float_scale);
}

}  // namespace

// Creates an instance of the TensorFlow dialect PrepareQuantize pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareQuantizePass(
    QuantizationMethod quantization_method) {
  return std::make_unique<PrepareQuantizePass>(quantization_method);
}

static PassRegistration<PrepareQuantizePass> pass;

}  // namespace quant
}  // namespace mlir
