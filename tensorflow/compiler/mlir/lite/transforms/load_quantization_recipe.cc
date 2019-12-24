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

// This transformation pass prepare the tflite fused ops for quantization.

#include "absl/memory/memory.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

//===----------------------------------------------------------------------===//
// The LoadQuantizationRecipe Pass.
//
namespace mlir {
namespace TFL {

namespace {

// This pass loads the quantization recipe for the TFLite ops to be quantized.
// Specifically, it extends the fused ops with their internal implementation as
// op regions. Each ops in the region produces results with element type
// AnyQuantizedType, thus bitwidth, narrow_range, etc are included. The op also
// defines the op quantization traits, which are used to propagate the
// quantization parameters by the following passes.
struct LoadQuantizationRecipe : public FunctionPass<LoadQuantizationRecipe> {
  void runOnFunction() override;

 private:
  void Initialize(LSTMOp lstm, OpBuilder* builder);

  // Create LSTM gates with different weights for input, recurrent and
  // cell state, and also the layer normalization parameters.
  Operation* CreateGate(Location loc, Value in, Value in_w, Value rec,
                        Value rec_w,
                        llvm::Optional<std::pair<Value, Value>> cell,
                        Value ln_w, Value ln_bias, OpBuilder* builder);

  Operation* CreateLayerNorm(Location loc, Value in, Value ln_w, Value ln_bias,
                             OpBuilder* builder);

  // Add the internal implementation of the LSTM to its regions.
  void LoadForLSTMOp(LSTMOp lstm, OpBuilder* builder);

  StringAttr none_af;
  StringAttr fc_format;
  BoolAttr keep_dims;
  Type int8;
  Type int16;
  ConstantOp none_cst;
};

void LoadQuantizationRecipe::Initialize(LSTMOp lstm, OpBuilder* builder) {
  Type expressed_type =
      lstm.input()->getType().cast<ShapedType>().getElementType();
  Type int8_storage_type = builder->getIntegerType(8);
  Type int16_storage_type = builder->getIntegerType(16);
  auto flag = quant::QuantizationFlags::FlagValue::Signed;
  int64_t int8_min = quant::QuantizedType::getDefaultMinimumForInteger(
      flag, /*integralWidth=*/8);
  int64_t int8_max = quant::QuantizedType::getDefaultMaximumForInteger(
      flag, /*integralWidth=*/8);
  int64_t int16_min = quant::QuantizedType::getDefaultMinimumForInteger(
      flag, /*integralWidth=*/16);
  int64_t int16_max = quant::QuantizedType::getDefaultMaximumForInteger(
      flag, /*integralWidth=*/16);
  auto any_int8 = quant::AnyQuantizedType::get(
      flag, int8_storage_type, expressed_type, int8_min, int8_max);
  auto any_int16 = quant::AnyQuantizedType::get(
      flag, int16_storage_type, expressed_type, int16_min, int16_max);

  int8 = any_int8.castFromExpressedType(lstm.input()->getType());
  int16 = any_int16.castFromExpressedType(lstm.input()->getType());
}

Operation* LoadQuantizationRecipe::CreateLayerNorm(Location loc, Value in,
                                                   Value ln_w, Value ln_bias,
                                                   OpBuilder* builder) {
  // Note that l2_normalization and add ops here are not the execution kernel
  // implementation for layer_normalization and we just want to use them to
  // model the quantization requirement.
  auto l2_norm = builder->create<L2NormalizationOp>(loc, int16, in, none_af);
  auto add = builder->create<AddOp>(loc, int16, in, l2_norm, none_af);
  return builder->create<FullyConnectedOp>(loc, int16, add, ln_w, ln_bias,
                                           none_af, fc_format, keep_dims);
}

Operation* LoadQuantizationRecipe::CreateGate(
    Location loc, Value in, Value in_w, Value rec, Value rec_w,
    llvm::Optional<std::pair<Value, Value>> cell, Value ln_w, Value ln_bias,
    OpBuilder* builder) {
  auto s1 = builder->create<FullyConnectedOp>(loc, int16, in, in_w, none_cst,
                                              none_af, fc_format, keep_dims);
  auto s2 = builder->create<FullyConnectedOp>(loc, int16, rec, rec_w, none_cst,
                                              none_af, fc_format, keep_dims);

  AddNOp s4;
  if (cell.hasValue()) {
    auto s3 = builder->create<MulOp>(loc, int16, cell.getValue().first,
                                     cell.getValue().second, none_af);
    s4 = builder->create<AddNOp>(
        loc, int16,
        llvm::ArrayRef<Value>(
            {*s1.output().begin(), *s2.output().begin(), s3.output()}));

  } else {
    s4 = builder->create<AddNOp>(
        loc, int16,
        llvm::ArrayRef<Value>({*s1.output().begin(), *s2.output().begin()}));
  }

  auto s5 = CreateLayerNorm(loc, s4.sum(), ln_w, ln_bias, builder);

  if (cell.hasValue()) {
    return builder->create<LogisticOp>(loc, int16, s5->getResult(0));
  } else {
    return builder->create<TanhOp>(loc, int16, s5->getResult(0));
  }
}

void LoadQuantizationRecipe::LoadForLSTMOp(LSTMOp lstm, OpBuilder* builder) {
  Initialize(lstm, builder);

  Region region;
  region.push_back(new Block);
  builder->setInsertionPointToEnd(&region.front());
  Location loc = lstm.getLoc();
  none_cst = builder->create<ConstantOp>(loc, builder->getNoneType(),
                                         builder->getUnitAttr());

  auto input_gate = CreateGate(
      loc, lstm.input(), lstm.input_to_input_weights(),
      lstm.input_activation_state(), lstm.recurrent_to_input_weights(),
      llvm::Optional<std::pair<Value, Value>>(
          {lstm.input_cell_state(), lstm.cell_to_input_weights()}),
      lstm.input_layer_norm_coefficients(), lstm.input_gate_bias(), builder);

  auto forget_gate = CreateGate(
      loc, lstm.input(), lstm.input_to_forget_weights(),
      lstm.input_activation_state(), lstm.recurrent_to_forget_weights(),
      llvm::Optional<std::pair<Value, Value>>(
          {lstm.input_cell_state(), lstm.cell_to_forget_weights()}),
      lstm.forget_layer_norm_coefficients(), lstm.forget_gate_bias(), builder);

  auto cell_gate = CreateGate(loc, lstm.input(), lstm.input_to_cell_weights(),
                              lstm.input_activation_state(),
                              lstm.recurrent_to_cell_weights(), llvm::None,
                              lstm.cell_layer_norm_coefficients(),
                              lstm.cell_bias(), builder);

  auto forget_cell_state = builder->create<MulOp>(
      loc, int16, forget_gate->getResult(0), lstm.input_cell_state(), none_af);
  auto input_cell_state = builder->create<MulOp>(
      loc, int16, input_gate->getResult(0), cell_gate->getResult(0), none_af);
  auto new_cell = builder->create<AddOp>(loc, int16, forget_cell_state.output(),
                                         input_cell_state.output(), none_af);

  auto output_gate = CreateGate(
      loc, lstm.input(), lstm.input_to_output_weights(),
      lstm.input_activation_state(), lstm.recurrent_to_output_weights(),
      llvm::Optional<std::pair<Value, Value>>(
          {new_cell, lstm.cell_to_output_weights()}),
      lstm.output_layer_norm_coefficients(), lstm.output_gate_bias(), builder);

  auto new_cell_tanh = builder->create<TanhOp>(loc, int16, new_cell);
  auto hidden_state = builder->create<MulOp>(
      loc, int16, new_cell_tanh.y(), output_gate->getResult(0), none_af);
  auto act = builder->create<FullyConnectedOp>(
      loc, int8, hidden_state.output(), lstm.projection_weights(),
      lstm.projection_bias(), none_af, fc_format, keep_dims);

  // TODO(fengliuai): define and register the op in the QuantOps Dialect.
  OperationState return_state(loc, "tf_quant.pseudo_return", act.getResult(0),
                              {int8}, {});
  builder->createOperation(return_state);

  lstm.internal().takeBody(region);
}

void LoadQuantizationRecipe::runOnFunction() {
  FuncOp func = getFunction();
  OpBuilder builder(func);
  none_af = builder.getStringAttr("NONE");
  fc_format = builder.getStringAttr("DEFAULT");
  keep_dims = builder.getBoolAttr(false);

  func.walk([&](Operation* op) {
    if (auto lstm = llvm::dyn_cast<LSTMOp>(op)) {
      LoadForLSTMOp(lstm, &builder);
    }
    // Handles other ops.
  });
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect LoadQuantizationRecipe
// pass.
std::unique_ptr<OpPassBase<FuncOp>> CreateLoadQuantizationRecipePass() {
  return absl::make_unique<LoadQuantizationRecipe>();
}

static PassRegistration<LoadQuantizationRecipe> pass(
    "tfl-load-recipe", "Load TFL op quantization recipe");

}  // namespace TFL
}  // namespace mlir
