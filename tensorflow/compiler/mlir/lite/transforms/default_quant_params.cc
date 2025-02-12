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

#include <memory>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/transforms/prepare_quantize_helper.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/quantization/common/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/quantization/common/quantization_lib/quantization_utils.h"

//===----------------------------------------------------------------------===//
// The Pass to add default quantization parameters for the activations which
// don't have quantization information. These default parameters are usually
// not from real measurement, so this pass is only for test purpose.

namespace mlir {
namespace TFL {
// Includs an auto-generated function, which can retrieve the quantization
// specification for an TFL operation. The signature of the function is
//   std::unique_pointer<OpQuantSpec> TFL::GetOpQuantSpec(Operation *)
#include "tensorflow/compiler/mlir/lite/utils/generated_op_quant_spec_getters.inc"

namespace {

#define GEN_PASS_DEF_DEFAULTQUANTPARAMSPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

class DefaultQuantParamsPass
    : public impl::DefaultQuantParamsPassBase<DefaultQuantParamsPass> {
 public:
  using DefaultQuantParamsPassBase::DefaultQuantParamsPassBase;

  explicit DefaultQuantParamsPass(double default_min, double default_max,
                                  bool is_signed) {
    this->default_min_ = default_min;
    this->default_max_ = default_max;
    this->is_signed_ = is_signed;
  }

  explicit DefaultQuantParamsPass(
      const DefaultQuantParamsPassOptions &options) {
    this->default_min_ = options.default_min_;
    this->default_max_ = options.default_max_;
    this->is_signed_ = options.is_signed_;
  }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DefaultQuantParamsPass)
  void runOnOperation() override;

 private:
  // Whether the value is used as a bias input of another op. Here we assume
  // bias is used immediately by the user. This assumption is always correct
  // after constant folding.
  bool UsedAsBias(Value value) {
    for (auto &use : value.getUses()) {
      auto biases = TFL::GetOpQuantSpec(use.getOwner())->biases_params;
      if (biases.find(use.getOperandNumber()) != biases.end()) return true;
    }
    return false;
  }

  // Uses `quant_params` to quantize `value` and inserting a pair of
  // tfl.quantize and tfl.dequantize ops for this `value`.
  void QuantizeValue(OpBuilder builder, Value value,
                     quant::QuantParams quant_params);

  // If the value hasn't been quantized, the functions adds it to `values`.
  void AddToWorkListIfUnquantized(Value value, std::vector<Value> *values);

  // Converts the default min/max to the default quantization parameters.
  quant::QuantParams GetDefaultQuantParams(Builder builder);

  // Gets the quantization parameters for the bias of an operation by using the
  // quantization parameters from the non-biases operands.
  quant::QuantParams GetQuantParamsForBias(Operation *op, int bias,
                                           const std::vector<int> &non_biases,
                                           quant::AccumulatorScaleFunc func);
  quant::QuantParams default_quant_params_;
};
}  // namespace

void DefaultQuantParamsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder builder(func);

  std::vector<Value> activation_values;
  std::vector<Value> bias_values;

  // First of all, collect all the values (block arguments and op results) which
  // are required to be quantized.
  for (auto arg : func.getBody().begin()->getArguments()) {
    if (UsedAsBias(arg)) {
      AddToWorkListIfUnquantized(arg, &bias_values);
    } else {
      AddToWorkListIfUnquantized(arg, &activation_values);
    }
  }

  func.walk([&](Operation *op) {
    if (!quant::IsOpQuantizable(op) || op->getParentOfType<TFL::CustomTfOp>()) {
      return;
    }

    for (auto res : op->getResults()) {
      if (UsedAsBias(res)) {
        AddToWorkListIfUnquantized(res, &bias_values);
      } else {
        AddToWorkListIfUnquantized(res, &activation_values);
      }
    }
  });

  // Apply the default quantization parameters for these activation values.
  quant::QuantParams default_params = GetDefaultQuantParams(builder);
  for (Value value : activation_values) {
    QuantizeValue(builder, value, default_params);
  }

  // Since all the non-biases operands have quantization parameters now, we
  // should be able to propagate them to the bias operand.
  for (Value bias : bias_values) {
    Operation *op = *bias.user_begin();
    auto spec = TFL::GetOpQuantSpec(op);
    for (auto &it : spec->biases_params) {
      quant::QuantParams bias_params = GetQuantParamsForBias(
          op, it.first, it.second.first, it.second.second);
      if (!bias_params) continue;
      QuantizeValue(builder, bias, bias_params);
    }
  }
}

void DefaultQuantParamsPass::AddToWorkListIfUnquantized(
    Value value, std::vector<Value> *values) {
  // If the result isn't with float type, this result is an integer tensor and
  // doesn't require quantization.
  auto tensor_type = mlir::dyn_cast<TensorType>(value.getType());
  if (!tensor_type) {
    // There are none type values.
    return;
  }
  if (!tensor_type.getElementType().isF32()) return;

  // If the result is consumed by a quantize op, it has been quantized.
  if (value.hasOneUse() &&
      llvm::isa<TFL::QuantizeOp>(*value.getUsers().begin()))
    return;

  // Add this result to the list to apply the default value.
  values->push_back(value);
}

void DefaultQuantParamsPass::QuantizeValue(OpBuilder builder, Value value,
                                           quant::QuantParams quant_params) {
  Type expressed_type = value.getType();
  Type new_type = quant_params.castFromExpressedType(expressed_type);
  // This value isn't an expressed type (float), skip.
  if (!new_type) return;

  Block &block = value.getParentRegion()->front();
  Operation *op = value.getDefiningOp();
  if (op) {
    builder.setInsertionPoint(&block, ++Block::iterator(op));
  } else {
    builder.setInsertionPointToStart(&block);
  }
  TypeAttr type_attr = TypeAttr::get(new_type);
  auto quantize = builder.create<TFL::QuantizeOp>(value.getLoc(), new_type,
                                                  value, type_attr);
  auto dequantize = builder.create<TFL::DequantizeOp>(
      value.getLoc(), expressed_type, quantize.getOutput());
  value.replaceAllUsesWith(dequantize);

  // `quantize` is using `dequantize` now, so we should set its operand to
  // `value`.
  quantize.getOperation()->replaceUsesOfWith(dequantize, value);
}

quant::QuantParams DefaultQuantParamsPass::GetQuantParamsForBias(
    Operation *op, int bias, const std::vector<int> &non_biases,
    quant::AccumulatorScaleFunc func) {
  std::vector<quant::QuantizedType> non_bias_types;
  non_bias_types.reserve(non_biases.size());
  for (int non_bias : non_biases) {
    Operation *non_bias_define = op->getOperand(non_bias).getDefiningOp();
    if (auto dequant = llvm::dyn_cast<TFL::DequantizeOp>(non_bias_define)) {
      auto non_bias_type = mlir::cast<TensorType>(dequant.getInput().getType());
      auto non_bias_ele_type =
          mlir::cast<quant::QuantizedType>(non_bias_type.getElementType());
      non_bias_types.push_back(non_bias_ele_type);
    } else {
      // The non-bias hasn't been quantized, let's skip this bias.
      break;
    }
  }
  // The non-bias hasn't been quantized, let's skip this bias.
  if (non_bias_types.size() != non_biases.size()) return {};

  return func(/*op_types=*/non_bias_types, /*adjusted_quant_dim=*/-1,
              /*legacy_float_scale=*/false);
}

quant::QuantParams DefaultQuantParamsPass::GetDefaultQuantParams(
    Builder builder) {
  if (!default_quant_params_) {
    default_quant_params_ = quantfork::fakeQuantAttrsToType(
        builder.getUnknownLoc(),
        /*numBits=*/8, default_min_, default_max_, /*narrowRange=*/false,
        builder.getF32Type(), is_signed_);
  }
  return default_quant_params_;
}

// Creates an instance of the default quant parameters pass.
std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantParamsPass(
    double default_min, double default_max, bool is_signed) {
  return std::make_unique<DefaultQuantParamsPass>(default_min, default_max,
                                                  is_signed);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantParamsPass(
    const DefaultQuantParamsPassOptions &options) {
  return std::make_unique<DefaultQuantParamsPass>(options);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateDefaultQuantParamsPass() {
  return std::make_unique<DefaultQuantParamsPass>();
}

}  // namespace TFL
}  // namespace mlir
