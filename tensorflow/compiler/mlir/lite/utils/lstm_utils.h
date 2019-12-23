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

// This header file defines common utils used by TFLite transformation
// passes to work with op attributes.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LSTM_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LSTM_UTILS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

constexpr char kTFImplements[] = "tf._implements";
constexpr char kLstmCellSimple[] = "LSTMCellSimple";
constexpr char kLayerNormalizedLstmCellSimple[] =
    "LayerNormalizedLstmCellSimple";
constexpr char kCoupleInputForgetGates[] = "CoupleInputForgetGates";

// A utility class that enables the conversion of the LSTMCellSimple composite
// op into a fused TFL LSTM op. The fused op is contained within a FuncOp
// that also contains other supporting ops needed to construct the operands for
// the fused op. The caller provides the containing FuncOp as input with
// arguments specifying the input, weight, projection and bias.
// The weight, projection, bias and layer norm scale all need to be
// RankedTensorType.
// This class sets the layer norm coefficients to NoneType.
class ConvertLSTMCellSimpleToFusedLSTM {
 public:
  explicit ConvertLSTMCellSimpleToFusedLSTM(mlir::FuncOp fused_func_op)
      : fused_func_op_(fused_func_op),
        couple_input_forget_gates_(false),
        builder_(fused_func_op.getBody()) {}

  // not copyable.
  ConvertLSTMCellSimpleToFusedLSTM(const ConvertLSTMCellSimpleToFusedLSTM&) =
      delete;
  ConvertLSTMCellSimpleToFusedLSTM& operator=(
      const ConvertLSTMCellSimpleToFusedLSTM&) = delete;
  virtual ~ConvertLSTMCellSimpleToFusedLSTM() {}

  virtual llvm::StringRef GetCompositeOpName() { return kLstmCellSimple; }

  // Rewrite the func body with constructed fused lstm.
  LogicalResult RewriteFunc();

  int GetNumInputs() { return n_input_; }

 protected:
  // verify input func op arguments/attributes and initialize internal state.
  virtual LogicalResult InitializeFromFuncAttributes();
  virtual LogicalResult Initialize();

  void UpdateFuncSignature();
  void GenerateFusedOpOperands();

  void SetWeightForInputToCellGate();
  void SetWeightForInputToInputGate();
  void SetWeightForInputToForgetGate();
  void SetWeightForInputToOutputGate();

  void SetWeightForRecurrentToCellGate();
  void SetWeightForRecurrentToInputGate();
  void SetWeightForRecurrentToForgetGate();
  void SetWeightForRecurrentToOutputGate();

  void SetBiasToCellGate();
  void SetBiasToInputGate();
  void SetBiasToForgetGate();
  void SetBiasToOutputGate();

  void SetProjection();
  void SetProjectionBias();

  void SetInputActivationState();
  void SetInputCellState();

  virtual void SetCellLayerNormCoefficients();
  virtual void SetInputLayerNormCoefficients();
  virtual void SetForgetLayerNormCoefficients();
  virtual void SetOutputLayerNormCoefficients();

  // specified state
  FuncOp fused_func_op_;
  ValuePtr input_;
  ValuePtr weight_;
  ValuePtr bias_;
  ValuePtr projection_;
  bool couple_input_forget_gates_;

  // internal state
  ValuePtr weight_transposed_;
  ValuePtr projection_transposed_;
  RankedTensorType weight_type_;
  RankedTensorType projection_type_;
  int num_gates_;
  int n_cell_;
  int n_output_;
  int n_input_;
  int num_cols_weight_transposed_;
  int num_cols_projection_transposed_;

  // input -> cifg
  ValuePtr input2input_;
  ValuePtr input2forget_;
  ValuePtr input2cell_;
  ValuePtr input2output_;

  // recurrent -> cifg
  ValuePtr rec2input_;
  ValuePtr rec2forget_;
  ValuePtr rec2cell_;
  ValuePtr rec2output_;

  // bias -> cifg
  ValuePtr bias2input_;
  ValuePtr bias2forget_;
  ValuePtr bias2cell_;
  ValuePtr bias2output_;

  // projection
  ValuePtr proj_weight_;
  ValuePtr proj_bias_;

  // state
  ValuePtr input_activation_state_;
  ValuePtr input_cell_state_;

  // layer norm coefficients
  ValuePtr input_layer_norm_coefficients_;
  ValuePtr forget_layer_norm_coefficients_;
  ValuePtr cell_layer_norm_coefficients_;
  ValuePtr output_layer_norm_coefficients_;

  mlir::TFL::LSTMOp lstm_;

  ValuePtr none_;
  SmallVector<int64_t, 1> bias_slice_shape_;
  SmallVector<int64_t, 1> bias_size_values_;
  SmallVector<int64_t, 2> weight_slice_shape_;
  SmallVector<int64_t, 2> weight_slice_size_input_values_;
  SmallVector<int64_t, 2> weight_slice_size_recurrent_values_;
  OpBuilder builder_;
};

// A utility class that enables the conversion of the
// LayerNormalizedLSTMCellSimple composite op into a fused TFL LSTM op. The
// fused op is contained within a FuncOp that also contains other supporting ops
// needed to construct the operands for the fused op. The caller provides the
// containing FuncOp as input with arguments specifying the input, weight,
// projection, bias and layer norm scale. The weight, projection, bias and
// layer norm scale all need to be RankedTensorType.
// This class overrides the layer norm coefficient setters from the base class.
class ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM
    : public ConvertLSTMCellSimpleToFusedLSTM {
 public:
  explicit ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM(
      mlir::FuncOp fused_func_op)
      : ConvertLSTMCellSimpleToFusedLSTM(fused_func_op) {}

  // not copyable.
  ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM(
      const ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM&) = delete;
  ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM& operator=(
      const ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM&) = delete;
  ~ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM() override {}

  llvm::StringRef GetCompositeOpName() override {
    return kLayerNormalizedLstmCellSimple;
  }

 protected:
  LogicalResult Initialize() override;

  void SetCellLayerNormCoefficients() override;
  void SetInputLayerNormCoefficients() override;
  void SetForgetLayerNormCoefficients() override;
  void SetOutputLayerNormCoefficients() override;

 private:
  // specified state
  ValuePtr layer_norm_scale_;

  // internal state
  RankedTensorType layer_norm_scale_type_;
  SmallVector<int64_t, 1> layer_norm_slice_shape_;
  SmallVector<int64_t, 1> layer_norm_size_values_;
};

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LSTM_UTILS_H_
