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

#include "tensorflow/compiler/mlir/lite/utils/lstm_utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Identifier.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/OpDefinition.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

Value* CreateI32SplatConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                           int32_t val, mlir::Location location) {
  auto type = builder->getTensorType(shape, builder->getIntegerType(32));
  auto attr = DenseElementsAttr::get(type, val);
  return builder->create<ConstantOp>(location, type, attr);
}

Value* CreateF32SplatConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                           float val, mlir::Location location) {
  auto type = builder->getTensorType(shape, builder->getF32Type());
  auto attr = DenseElementsAttr::get(type, val);
  return builder->create<ConstantOp>(location, type, attr);
}

Value* CreateI64DenseConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                           ArrayRef<int64_t> values, mlir::Location location) {
  auto type = builder->getTensorType(static_cast<int>(shape.size()),
                                     builder->getIntegerType(64));
  auto attr = DenseElementsAttr::get(type, values);
  return builder->create<ConstantOp>(location, type, attr);
}

Value* CreateNoneValue(OpBuilder* builder, mlir::Location location) {
  return builder->create<mlir::ConstantOp>(location, builder->getNoneType(),
                                           builder->getUnitAttr());
}

Value* Transpose2D(OpBuilder* builder, Value* value_to_transpose,
                   RankedTensorType type, mlir::Location location) {
  // Create a constant op for transpose permutation.
  SmallVector<int64_t, 2> perm = {1, 0};
  auto perm_op = CreateI64DenseConst(builder, perm, perm, location);

  // Create tensor type for the transpose result.
  auto transpose_type = type;
  auto transpose_shape = functional::map(
      [transpose_type](int64_t dim) { return transpose_type.getDimSize(dim); },
      perm);
  auto elem_type = transpose_type.getElementType();
  auto result_type = builder->getTensorType(transpose_shape, elem_type);

  return builder->create<TF::TransposeOp>(location, result_type,
                                          value_to_transpose, perm_op);
}

Value* SliceRankedTensor(OpBuilder* builder, Value* input,
                         ArrayRef<int64_t> begin_shape,
                         ArrayRef<int64_t> begin_values,
                         ArrayRef<int64_t> size_shape,
                         ArrayRef<int64_t> size_values,
                         mlir::Location location) {
  // Create a dense constant op for slice's begin
  auto slice_i2c_begin =
      CreateI64DenseConst(builder, begin_shape, begin_values, location);

  // Create a dense constant op for slice's size
  auto slice_i2c_size =
      CreateI64DenseConst(builder, size_shape, size_values, location);

  return builder->create<TF::SliceOp>(
      location,
      builder->getTensorType(
          size_values,
          input->getType().cast<RankedTensorType>().getElementType()),
      input, slice_i2c_begin, slice_i2c_size);
}

}  // namespace

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToCellGate() {
  SmallVector<int64_t, 2> begin_i2c_values = {0, 0};
  input2cell_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_i2c_values,
      weight_slice_shape_, weight_slice_size_input_values_,
      fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToInputGate() {
  SmallVector<int64_t, 2> begin_i2i_values = {n_cell_, 0};
  input2input_ = couple_input_forget_gates_
                     ? none_
                     : SliceRankedTensor(&builder_, weight_transposed_,
                                         weight_slice_shape_, begin_i2i_values,
                                         weight_slice_shape_,
                                         weight_slice_size_input_values_,
                                         fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToForgetGate() {
  int input_forget_start = couple_input_forget_gates_ ? n_cell_ : 2 * n_cell_;
  SmallVector<int64_t, 2> begin_i2f_values = {input_forget_start, 0};
  input2forget_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_i2f_values,
      weight_slice_shape_, weight_slice_size_input_values_,
      fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForInputToOutputGate() {
  int input_output_start =
      couple_input_forget_gates_ ? 2 * n_cell_ : 3 * n_cell_;
  SmallVector<int64_t, 2> begin_i2o_values = {input_output_start, 0};
  input2output_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_i2o_values,
      weight_slice_shape_, weight_slice_size_input_values_,
      fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToCellGate() {
  SmallVector<int64_t, 2> begin_rec2c_values = {0, n_input_};
  rec2cell_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_rec2c_values,
      weight_slice_shape_, weight_slice_size_recurrent_values_,
      fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToInputGate() {
  SmallVector<int64_t, 2> begin_rec2i_values = {n_cell_, n_input_};
  rec2input_ = couple_input_forget_gates_
                   ? none_
                   : SliceRankedTensor(&builder_, weight_transposed_,
                                       weight_slice_shape_, begin_rec2i_values,
                                       weight_slice_shape_,
                                       weight_slice_size_recurrent_values_,
                                       fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToForgetGate() {
  int rec_forget_start = couple_input_forget_gates_ ? n_cell_ : 2 * n_cell_;
  SmallVector<int64_t, 2> begin_rec2f_values = {rec_forget_start, n_input_};
  rec2forget_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_rec2f_values,
      weight_slice_shape_, weight_slice_size_recurrent_values_,
      fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetWeightForRecurrentToOutputGate() {
  int rec_output_start = couple_input_forget_gates_ ? 2 * n_cell_ : 3 * n_cell_;
  SmallVector<int64_t, 2> begin_rec2o_values = {rec_output_start, n_input_};
  rec2output_ = SliceRankedTensor(
      &builder_, weight_transposed_, weight_slice_shape_, begin_rec2o_values,
      weight_slice_shape_, weight_slice_size_recurrent_values_,
      fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToCellGate() {
  SmallVector<int64_t, 1> begin_bias2c_values = {0};
  bias2cell_ = SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                                 begin_bias2c_values, bias_slice_shape_,
                                 bias_size_values_, fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToInputGate() {
  SmallVector<int64_t, 1> begin_bias2i_values = {n_cell_};
  bias2input_ =
      couple_input_forget_gates_
          ? none_
          : SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                              begin_bias2i_values, bias_slice_shape_,
                              bias_size_values_, fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToForgetGate() {
  int bias_forget_start = couple_input_forget_gates_ ? n_cell_ : 2 * n_cell_;
  SmallVector<int64_t, 1> begin_bias2f_values = {bias_forget_start};
  bias2forget_ = SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                                   begin_bias2f_values, bias_slice_shape_,
                                   bias_size_values_, fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetBiasToOutputGate() {
  int bias_output_start =
      couple_input_forget_gates_ ? 2 * n_cell_ : 3 * n_cell_;
  SmallVector<int64_t, 1> begin_bias2o_values = {bias_output_start};
  bias2output_ = SliceRankedTensor(&builder_, bias_, bias_slice_shape_,
                                   begin_bias2o_values, bias_slice_shape_,
                                   bias_size_values_, fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetProjection() {
  SmallVector<int64_t, 2> projection_slice_shape = {
      1, num_cols_projection_transposed_};
  SmallVector<int64_t, 2> projection_slice_size_values = {n_output_, n_cell_};
  SmallVector<int64_t, 2> projection_slice_begin_values = {0, 0};
  proj_weight_ =
      !projection_
          ? none_
          : SliceRankedTensor(
                &builder_, projection_transposed_, projection_slice_shape,
                projection_slice_begin_values, projection_slice_shape,
                projection_slice_size_values, fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetProjectionBias() {
  proj_bias_ = !projection_type_
                   ? none_
                   : CreateF32SplatConst(&builder_, {n_output_}, 0,
                                         fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetInputActivationState() {
  input_activation_state_ = CreateF32SplatConst(&builder_, {1, n_output_}, 0,
                                                fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetInputCellState() {
  input_cell_state_ =
      CreateF32SplatConst(&builder_, {1, n_cell_}, 0, fused_func_op_.getLoc());
}

void ConvertLSTMCellSimpleToFusedLSTM::SetCellLayerNormCoefficients() {
  cell_layer_norm_coefficients_ = none_;
}

void ConvertLSTMCellSimpleToFusedLSTM::SetInputLayerNormCoefficients() {
  input_layer_norm_coefficients_ = none_;
}

void ConvertLSTMCellSimpleToFusedLSTM::SetForgetLayerNormCoefficients() {
  forget_layer_norm_coefficients_ = none_;
}
void ConvertLSTMCellSimpleToFusedLSTM::SetOutputLayerNormCoefficients() {
  output_layer_norm_coefficients_ = none_;
}

void ConvertLSTMCellSimpleToFusedLSTM::GenerateFusedOpOperands() {
  // Transpose both weight and projection.
  weight_transposed_ =
      Transpose2D(&builder_, weight_, weight_type_, fused_func_op_.getLoc());
  projection_transposed_ = Transpose2D(&builder_, projection_, projection_type_,
                                       fused_func_op_.getLoc());

  none_ = CreateNoneValue(&builder_, fused_func_op_.getLoc());
  // Extract input to cifg gates via slicing the weight tensor
  SetWeightForInputToCellGate();
  SetWeightForInputToInputGate();
  SetWeightForInputToForgetGate();
  SetWeightForInputToOutputGate();

  // Extract recurrent to cifg gates via slicing the weight tensor
  SetWeightForRecurrentToCellGate();
  SetWeightForRecurrentToInputGate();
  SetWeightForRecurrentToForgetGate();
  SetWeightForRecurrentToOutputGate();

  // Extract bias to cifg gates via slicing the bias tensor
  SetBiasToCellGate();
  SetBiasToInputGate();
  SetBiasToForgetGate();
  SetBiasToOutputGate();

  // Extract projection and set an empty projection bias
  SetProjection();
  SetProjectionBias();

  // Set the variable tensors
  SetInputActivationState();
  SetInputCellState();

  // Extract the layer norm coefficients
  SetCellLayerNormCoefficients();
  SetInputLayerNormCoefficients();
  SetForgetLayerNormCoefficients();
  SetOutputLayerNormCoefficients();
}

void ConvertLSTMCellSimpleToFusedLSTM::UpdateFuncSignature() {
  // https://github.com/tensorflow/community/pull/113
  auto attr = fused_func_op_.getAttrOfType<StringAttr>("tf_.implements");
  if (!attr) {
    fused_func_op_.setAttr("tf._implements",
                           builder_.getStringAttr(GetCompositeOpName()));
  }
  SmallVector<int64_t, 2> output_shape{1, -1};
  auto input_types = fused_func_op_.getType().getInputs();
  auto output_type = builder_.getTensorType(
      output_shape,
      input_->getType().cast<RankedTensorType>().getElementType());
  fused_func_op_.setType(mlir::FunctionType::get(input_types, output_type,
                                                 fused_func_op_.getContext()));
}

void ConvertLSTMCellSimpleToFusedLSTM::RewriteFunc() {
  // Update the func signature, based on output shape.
  // The func will ultimately return the output of the fused
  // LSTM op.
  UpdateFuncSignature();

  // Transform the weights, projection, bias and layer norm coefficients
  // to generate operands for the TFL fused LSTM op.
  GenerateFusedOpOperands();

  // Create the fused LSTM op.
  SmallVector<int64_t, 2> output_shape = {1, n_output_};
  auto result_type = builder_.getTensorType(
      output_shape,
      input_->getType().cast<RankedTensorType>().getElementType());
  lstm_ = builder_.create<mlir::TFL::LSTMOp>(
      fused_func_op_.getLoc(), result_type, input_, input2input_, input2forget_,
      input2cell_, input2output_, rec2input_, rec2forget_, rec2cell_,
      rec2output_, /*cell_to_input_weights*/ none_,
      /*cell_to_forget_weights*/ none_,
      /*cell_to_output_weights*/ none_, bias2input_, bias2forget_, bias2cell_,
      bias2output_, proj_weight_, proj_bias_, input_activation_state_,
      input_cell_state_, input_layer_norm_coefficients_,
      forget_layer_norm_coefficients_, cell_layer_norm_coefficients_,
      output_layer_norm_coefficients_, builder_.getStringAttr("TANH"),
      builder_.getF32FloatAttr(10.0), builder_.getF32FloatAttr(0.0),
      builder_.getStringAttr("FULL"));

  // Cast the static shaped lstm result to FuncOp's signature -
  // Ranked but unknown 2nd dimension to support stacking these.
  SmallVector<int64_t, 2> func_output_shape = {1, -1};
  auto func_result_type = builder_.getTensorType(
      func_output_shape,
      input_->getType().cast<RankedTensorType>().getElementType());

  auto tensor_cast = builder_.create<mlir::TensorCastOp>(
      fused_func_op_.getLoc(), lstm_.getResult(), func_result_type);
  builder_.create<mlir::ReturnOp>(fused_func_op_.getLoc(),
                                  tensor_cast.getResult());
}

LogicalResult ConvertLSTMCellSimpleToFusedLSTM::Initialize() {
  num_gates_ = couple_input_forget_gates_ ? 3 : 4;

  input_ = fused_func_op_.getArgument(0);
  bias_ = fused_func_op_.getArgument(2);

  weight_ = fused_func_op_.getArgument(1);
  weight_type_ = weight_->getType().cast<RankedTensorType>();

  if (weight_type_.getRank() != 2) {
    return fused_func_op_.emitError() << "The weight tensor was not of rank 2";
  }

  if (weight_type_.getDimSize(1) % num_gates_ != 0) {
    return fused_func_op_.emitError()
           << "Invalid dimension 1 of weight tensor, "
              "should be divisible by the number of gates";
  }
  n_cell_ = weight_type_.getDimSize(1) / num_gates_;

  projection_ = fused_func_op_.getArgument(3);
  projection_type_ = projection_->getType().cast<RankedTensorType>();
  if (projection_type_.getRank() != 2) {
    n_output_ = n_cell_;
  } else {
    n_output_ = projection_type_.getDimSize(1);
  }
  n_input_ = weight_type_.getDimSize(0) - n_output_;
  num_cols_weight_transposed_ = weight_type_.getDimSize(0);
  num_cols_projection_transposed_ = projection_type_.getDimSize(0);

  bias_slice_shape_ = {n_cell_};
  bias_size_values_ = {n_cell_};
  weight_slice_shape_ = {1, num_cols_weight_transposed_};
  weight_slice_size_input_values_ = {n_cell_, n_input_};
  weight_slice_size_recurrent_values_ = {n_cell_, n_output_};

  return success();
}

LogicalResult ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::Initialize() {
  if (failed(ConvertLSTMCellSimpleToFusedLSTM::Initialize())) {
    return fused_func_op_.emitError()
           << "Specified LayerNormalizedLSTMCellSimple was not of the expected "
              "interface and cannot not be converted to the fused LSTM op";
  }

  layer_norm_scale_ = fused_func_op_.getArgument(4);
  layer_norm_scale_type_ =
      layer_norm_scale_->getType().cast<RankedTensorType>();
  if (layer_norm_scale_type_.getRank() != 1) {
    return fused_func_op_.emitError()
           << "The layer_norm_scale tensor was not of rank 1";
  }
  layer_norm_slice_shape_ = {n_cell_};
  layer_norm_size_values_ = {n_cell_};

  return success();
}

void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetCellLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_cell_layer_norm_values = {0};
  cell_layer_norm_coefficients_ =
      SliceRankedTensor(&builder_, layer_norm_scale_, layer_norm_slice_shape_,
                        begin_cell_layer_norm_values, layer_norm_slice_shape_,
                        layer_norm_size_values_, fused_func_op_.getLoc());
}

void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetInputLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_input_layer_norm_values = {n_cell_};
  input_layer_norm_coefficients_ =
      couple_input_forget_gates_
          ? none_
          : SliceRankedTensor(
                &builder_, layer_norm_scale_, layer_norm_slice_shape_,
                begin_input_layer_norm_values, layer_norm_slice_shape_,
                layer_norm_size_values_, fused_func_op_.getLoc());
}

void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetForgetLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_forget_layer_norm_values = {2 * n_cell_};
  forget_layer_norm_coefficients_ =
      SliceRankedTensor(&builder_, layer_norm_scale_, layer_norm_slice_shape_,
                        begin_forget_layer_norm_values, layer_norm_slice_shape_,
                        layer_norm_size_values_, fused_func_op_.getLoc());
}

void ConvertLayerNormalizedLSTMCellSimpleToFusedLSTM::
    SetOutputLayerNormCoefficients() {
  SmallVector<int64_t, 1> begin_output_layer_norm_values = {3 * n_cell_};
  output_layer_norm_coefficients_ =
      SliceRankedTensor(&builder_, layer_norm_scale_, layer_norm_slice_shape_,
                        begin_output_layer_norm_values, layer_norm_slice_shape_,
                        layer_norm_size_values_, fused_func_op_.getLoc());
}

}  // namespace TFL
}  // namespace mlir
