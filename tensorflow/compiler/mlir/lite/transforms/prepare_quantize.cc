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
#include <iterator>
#include <string>

#include "absl/memory/memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/core/framework/types.pb.h"

// NOLINTNEXTLINE
static llvm::cl::list<std::string> quantize_whitelist(
    "tfl-test-quantize-whitelist", llvm::cl::value_desc("list"),
    llvm::cl::desc("comma separated list of whitelisted functions to be "
                   "quantized. Only used in tests"),
    llvm::cl::CommaSeparated);

// NOLINTNEXTLINE
static llvm::cl::opt<bool> quantize_signed(
    "tfl-test-quantize-signed", llvm::cl::value_desc("bool"),
    llvm::cl::desc("signed inference type. Only used in tests"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
static llvm::cl::opt<bool> disable_per_channel(
    "tfl-disable-per-channel", llvm::cl::value_desc("bool"),
    llvm::cl::desc("Whether disable per-channel quantized weights."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// The prepare-quantize Pass.
//
namespace mlir {
namespace TFL {

namespace {

// Applies prepare quantization on the model in TFL dialect. This pass runs
// before the quantization pass and propagate the quantization parameters
// across ops. This step is necessary for post-training quantization and also
// making the quantization rule for some operations in the quantization-aware
// training quantization simpler.
class PrepareQuantizePass : public FunctionPass<PrepareQuantizePass> {
 public:
  // Constructor used by the PassRegistration and enforce uint8 quantization.
  explicit PrepareQuantizePass() {
    if (quantize_signed)
      quant_specs_.inference_type = tensorflow::DT_QINT8;
    else
      quant_specs_.inference_type = tensorflow::DT_QUINT8;
  }

  // Constructor used by manually creating the pass.
  explicit PrepareQuantizePass(const QuantizationSpecs& quant_specs)
      : quant_specs_(quant_specs) {}

  void runOnFunction() override;

 private:
  // Set the quantization parameters of the input nodes. These parameters are
  // converted from the user specified input value ranges. The input nodes with
  // non-float tensor types will be skipped because they are not quantizable.
  // Return true if number of input nodes doesn't equal to that of the input
  // ranges.
  bool SetInputNodesQuantizationParams(FuncOp func);

  // The function might contain more stats ops than required, and it will
  // introduce requantize if the calibration stats have conflicts. This method
  // tries to remove all the redundant stats ops.
  bool RemoveRedundantStats(FuncOp func);

  // Verify the quantization specification is expected for quantizing the
  // current function.
  bool IsLegalQuantSpecs(FuncOp func) {
    if (func.getName() == quant_specs_.target_func) {
      return func.getNumArguments() == quant_specs_.input_ranges.size();
    }
    return true;
  }

  // Get the min and max values from the quantization specification for the
  // current function function and argument index. Uses default values if
  // the function is specified in the `quantize_whitelist`.
  std::pair<double, double> GetMinMaxValuesForArgument(
      llvm::StringRef func_name, int index) {
    if (func_name == quant_specs_.target_func) {
      return quant_specs_.input_ranges[index];
    } else {
      return {0.0, 255.0};
    }
  }

  QuantizationSpecs quant_specs_;
};

bool PrepareQuantizePass::SetInputNodesQuantizationParams(FuncOp func) {
  StringRef func_name = func.getName();
  auto& target_func = quant_specs_.target_func;

  // Skip this function because it isn't the target function from the spec or
  // in the function while list.
  if (target_func != func_name &&
      !llvm::is_contained(quantize_whitelist, func_name)) {
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
        auto min_max = GetMinMaxValuesForArgument(func_name, i);
        TypeAttr params = quant::GetQuantizedTypeAttr(
            builder, input_type, builder.getF64FloatAttr(min_max.first),
            builder.getF64FloatAttr(min_max.second), /*quant_dim=*/-1, num_bits,
            narrow_range, is_signed);
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

#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

bool PrepareQuantizePass::RemoveRedundantStats(FuncOp func) {
  return RemoveRedundantStatsOps(func, GetOpQuantSpec);
}

using PrepareQuantStats =
    quant::ConvertStatsToQDQs<quant::QuantizeCastOp, quant::DequantizeCastOp>;

void PrepareQuantizePass::runOnFunction() {
  FuncOp func = getFunction();
  MLIRContext* ctx = func.getContext();

  ConvertTFLQuantOpsToMlirQuantOps(func);

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

  // During the legalization, unsigned quantized type is used, so we have to
  // convert all of them to signed.
  OwningRewritePatternList patterns;
  bool is_signed = quant_specs_.IsSignedInferenceType();
  if (is_signed) {
    patterns.insert<quant::ConvertUnsignedToSigned<quant::QuantizeCastOp>>(ctx);
    // Convert quant stats to int8 quantization parameters.
    // Currently, only activation stats are imported, so narrow_range = false.
    patterns.insert<PrepareQuantStats>(8, false, true, ctx);
  } else {
    // Convert quant stats to uint8 quantization parameters.
    // Currently, only activation stats are imported, so narrow_range = false.
    patterns.insert<PrepareQuantStats>(8, false, false, ctx);
  }
  applyPatternsGreedily(func, patterns);

  // Finally, the quantization parameters can be propagated to the rest of the
  // values (tensors).
  ApplyQuantizationParamsPropagation(func, is_signed, disable_per_channel,
                                     GetOpQuantSpec);

  ConvertMlirQuantOpsToTFLQuantOps(func);
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect PrepareQuantize pass.
std::unique_ptr<OpPassBase<FuncOp>> CreatePrepareQuantizePass(
    const QuantizationSpecs& quant_specs) {
  return std::make_unique<PrepareQuantizePass>(quant_specs);
}

static PassRegistration<PrepareQuantizePass> pass(
    "tfl-prepare-quantize", "Prepare TFL dialect for quantization");

}  // namespace TFL
}  // namespace mlir
