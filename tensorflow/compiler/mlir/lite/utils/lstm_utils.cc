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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Identifier.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

namespace {

Value CreateI32SplatConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                          int32_t val, mlir::Location location) {
  auto type = RankedTensorType::get(shape, builder->getIntegerType(32));
  auto attr = DenseElementsAttr::get(type, val);
  return builder->create<ConstantOp>(location, type, attr);
}

Value CreateF32SplatConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                          float val, mlir::Location location) {
  auto type = RankedTensorType::get(shape, builder->getF32Type());
  auto attr = DenseElementsAttr::get(type, val);
  return builder->create<ConstantOp>(location, type, attr);
}

Value CreateI64DenseConst(OpBuilder* builder, ArrayRef<int64_t> shape,
                          ArrayRef<int64_t> values, mlir::Location location) {
  auto type = RankedTensorType::get(static_cast<int>(shape.size()),
                                    builder->getIntegerType(64));
  auto attr = DenseElementsAttr::get(type, values);
  return builder->create<ConstantOp>(location, type, attr);
}

Value CreateNoneValue(OpBuilder* builder, mlir::Location location) {
  return builder->create<mlir::ConstantOp>(location, builder->getNoneType(),
                                           builder->getUnitAttr());
}

Value Transpose2D(OpBuilder* builder, Value value_to_transpose,
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
  auto result_type = RankedTensorType::get(transpose_shape, elem_type);

  return builder->create<TF::TransposeOp>(location, result_type,
                                          value_to_transpose, perm_op);
}

ArrayRef<int64_t> GetRankedTensorShape(Value value) {
  return value.getType().cast<RankedTensorType>().getShape();
}

Value SliceRankedTensor(OpBuilder* builder, Value input,
                        ArrayRef<int64_t> begin_shape,
                        ArrayRef<int64_t> begin_values,
                        ArrayRef<int64_t> size_shape,
                        ArrayRef<int64_t> size_values,
                        mlir::Location location) {
  // If the size of the tensor to be sliced from the input overflows
  // the input tensor's dimensions, return 0-valued tensor of the requested
  // shape.
  ArrayRef<int64_t> input_shape = GetRankedTensorShape(input);
  for (int i = 0; i < input_shape.size(); i++) {
    if (begin_values[i] < 0 ||
        (begin_values[i] + size_values[i] > input_shape[i])) {
      return CreateF32SplatConst(builder, size_shape, 0, location);
    }
  }

  // Create a dense constant op for slice's begin
  auto slice_i2c_begin =
      CreateI64DenseConst(builder, begin_shape, begin_values, location);

  // Create a dense constant op for slice's size
  auto slice_i2c_size =
      CreateI64DenseConst(builder, size_shape, size_values, location);

  return builder->create<TF::SliceOp>(
      location,
      RankedTensorType::get(
          size_values,
          input.getType().cast<RankedTensorType>().getElementType()),
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
  SmallVector<int64_t, 2> output_shape{1, -1};
  auto input_types = fused_func_op_.getType().getInputs();
  auto output_type = mlir::RankedTensorType::get(
      output_shape, input_.getType().cast<RankedTensorType>().getElementType());
  fused_func_op_.setType(mlir::FunctionType::get(input_types, output_type,
                                                 fused_func_op_.getContext()));
}

LogicalResult ConvertLSTMCellSimpleToFusedLSTM::RewriteFunc() {
  LogicalResult result = Initialize();
  if (failed(result)) {
    return result;
  }

  // Update the func signature, based on output shape.
  // The func will ultimately return the output of the fused
  // LSTM op.
  UpdateFuncSignature();

  // Transform the weights, projection, bias and layer norm coefficients
  // to generate operands for the TFL fused LSTM op.
  GenerateFusedOpOperands();

  // Create the fused LSTM op.
  SmallVector<int64_t, 2> output_shape = {1, n_output_};
  auto result_type = mlir::RankedTensorType::get(
      output_shape, input_.getType().cast<RankedTensorType>().getElementType());
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
  auto func_result_type = mlir::RankedTensorType::get(
      func_output_shape,
      input_.getType().cast<RankedTensorType>().getElementType());

  auto tensor_cast = builder_.create<mlir::TensorCastOp>(
      fused_func_op_.getLoc(), lstm_.getResult(), func_result_type);
  builder_.create<mlir::ReturnOp>(fused_func_op_.getLoc(),
                                  tensor_cast.getResult());
  return success();
}

LogicalResult ConvertLSTMCellSimpleToFusedLSTM::InitializeFromFuncAttributes() {
  auto attr = fused_func_op_.getAttrOfType<StringAttr>(kTFImplements);
  if (!attr) {
    return fused_func_op_.emitError()
           << "Invalid function attribute, expected " << kTFImplements
           << " attribute "
              "not found";
  }

  // TODO(ashwinm, b/144775479): Make these NamedAttribute on TF import
  // once tf.function can support this.
  llvm::SmallVector<llvm::StringRef, 4> attr_tokens;
  attr.getValue().split(attr_tokens, ",");
  if (attr_tokens.empty()) {
    return fused_func_op_.emitError()
           << kTFImplements << " attribute should be set";
  }

  // Check if the interface matches.
  if (GetCompositeOpName().str() != attr_tokens[0]) {
    return fused_func_op_.emitError()
           << "Unexpected interface for the composite op. Expected: "
           << GetCompositeOpName() << " Actual: " << attr_tokens[0];
  }

  // Extract other interface attributes, for now cifg.
  couple_input_forget_gates_ =
      std::find(attr_tokens.begin() + 1, attr_tokens.end(),
                kCoupleInputForgetGates) != attr_tokens.end();

  return success();
}

LogicalResult ConvertLSTMCellSimpleToFusedLSTM::Initialize() {
  if (failed(InitializeFromFuncAttributes())) {
    return fused_func_op_.emitError()
           << "Expected function attributes were not set on the function "
              "encapsulating the composite op";
  }

  num_gates_ = couple_input_forget_gates_ ? 3 : 4;

  input_ = fused_func_op_.getArgument(0);
  bias_ = fused_func_op_.getArgument(2);

  weight_ = fused_func_op_.getArgument(1);
  weight_type_ = weight_.getType().cast<RankedTensorType>();

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
  projection_type_ = projection_.getType().cast<RankedTensorType>();
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
  layer_norm_scale_type_ = layer_norm_scale_.getType().cast<RankedTensorType>();
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

TF::ConstOp Create1DConstantOp(const std::vector<int>& value, Location loc,
                               OpBuilder* builder) {
  auto type =
      mlir::RankedTensorType::get(value.size(), builder->getIntegerType(32));
  auto dense_values = mlir::DenseIntElementsAttr::get(type, value);
  return builder->create<TF::ConstOp>(loc, dense_values);
}

TF::ConstOp CreateScalarConstantOp(int value, Location loc,
                                   OpBuilder* builder) {
  return builder->create<TF::ConstOp>(loc, builder->getI32IntegerAttr(value));
}

LogicalResult CreateEqualSizeSplitVOp(Value input, int axis, int splits,
                                      Location loc, OpBuilder* builder,
                                      Operation** result) {
  auto input_type = input.getType().cast<RankedTensorType>();
  SmallVector<int64_t, 4> output_shape;
  int size_of_splits;
  if (input_type.getRank() < axis || axis < 0) return failure();
  for (int i = 0; i < input_type.getRank(); ++i) {
    int dim = input_type.getDimSize(i);
    if (i == axis) {
      if (dim % splits != 0) {
        return failure();
      }
      size_of_splits = dim / splits;
      output_shape.push_back(size_of_splits);
    } else {
      output_shape.push_back(dim);
    }
  }

  SmallVector<mlir::Type, 4> output_types;
  for (int i = 0; i < splits; ++i) {
    output_types.push_back(
        mlir::RankedTensorType::get(output_shape, input_type.getElementType()));
  }
  auto size_of_splits_op = Create1DConstantOp(
      {size_of_splits, size_of_splits, size_of_splits, size_of_splits}, loc,
      builder);

  auto axis_op = CreateScalarConstantOp(axis, loc, builder);
  *result = builder->create<TF::SplitVOp>(loc, output_types, input,
                                          size_of_splits_op.getResult(),
                                          axis_op.getResult());
  return success();
}

void UpdateFuncSignature(int batch, int time, int output,
                         mlir::FuncOp* func_op) {
  SmallVector<int64_t, 4> output_shape{batch, time, output};
  auto input_types = func_op->getType().getInputs();
  auto element_type = input_types[0].cast<RankedTensorType>().getElementType();
  auto output_type = mlir::RankedTensorType::get(output_shape, element_type);
  func_op->setType(
      mlir::FunctionType::get(input_types, output_type, func_op->getContext()));
}

// TODO(b/147436982): Consider refactor this to be more general.
LogicalResult ConvertKerasLSTMLayer(mlir::FuncOp func_op, OpBuilder* builder) {
  // For argument order, please check out standard_lstm under
  // tensorflow/python/keras/layers/recurrent_v2.py
  Value input = func_op.getArgument(0);
  Value output_init_state = func_op.getArgument(1);
  Value hidden_init_state = func_op.getArgument(2);
  Value weight_kernel = func_op.getArgument(3);
  Value recurrent_kernel = func_op.getArgument(4);
  Value bias = func_op.getArgument(5);

  // Assume it's batch majored.
  auto input_type = input.getType().dyn_cast_or_null<RankedTensorType>();
  if (!input_type) {
    func_op.emitError() << "Input type is not a ranked tensor type";
    return failure();
  }

  int batch = input_type.getDimSize(0);
  int time = input_type.getDimSize(1);

  // Setup correct weights.
  RankedTensorType weight_type =
      weight_kernel.getType().cast<RankedTensorType>();
  if (weight_type.getRank() != 2)
    return func_op.emitError() << "The weight should be rank of 2";

  Value transposed_weight_kernel =
      Transpose2D(builder, weight_kernel, weight_type, func_op.getLoc());

  RankedTensorType recurrent_kernel_type =
      recurrent_kernel.getType().cast<RankedTensorType>();
  const int n_output = recurrent_kernel_type.getDimSize(0);

  Value transpose_recurrent_kernel = Transpose2D(
      builder, recurrent_kernel, recurrent_kernel_type, func_op.getLoc());

  // Splits the weights into 4: i, f, c, o.
  const int splits = 4;

  Operation* weights_array;
  if (failed(CreateEqualSizeSplitVOp(transposed_weight_kernel, 0, splits,
                                     func_op.getLoc(), builder,
                                     &weights_array)))
    return failure();

  // Splits the recurrent_weights into 4:
  Operation* recurrent_weights_array;
  if (failed(CreateEqualSizeSplitVOp(transpose_recurrent_kernel, 0, splits,
                                     func_op.getLoc(), builder,
                                     &recurrent_weights_array)))
    return failure();

  // Splits the bias into 4:
  Operation* bias_array;
  if (failed(CreateEqualSizeSplitVOp(bias, 0, splits, func_op.getLoc(), builder,
                                     &bias_array)))
    return failure();

  // Update the function signature:
  UpdateFuncSignature(batch, time, n_output, &func_op);

  // Build the lstm op.
  SmallVector<int64_t, 3> output_shape = {batch, time, n_output};
  auto result_type = mlir::RankedTensorType::get(
      output_shape, input.getType().cast<RankedTensorType>().getElementType());

  Value none = builder->create<mlir::ConstantOp>(
      func_op.getLoc(), builder->getNoneType(), builder->getUnitAttr());
  auto lstm = builder->create<mlir::TFL::LSTMOp>(
      func_op.getLoc(), result_type, /*input=*/input,
      /*input_to_input_weights=*/weights_array->getResult(0),
      /*input_to_forget_weights=*/weights_array->getResult(1),
      /*input_to_cell_weights=*/weights_array->getResult(2),
      /*input_to_output_weights=*/weights_array->getResult(3),
      /*recurrent_to_input_weights=*/recurrent_weights_array->getResult(0),
      /*recurrent_to_forget_weights=*/recurrent_weights_array->getResult(1),
      /*recurrent_to_cell_weights=*/recurrent_weights_array->getResult(2),
      /*recurrent_to_output_weights=*/recurrent_weights_array->getResult(3),
      /*cell_to_input_weights=*/none,
      /*cell_to_forget_weights=*/none,
      /*cell_to_output_weights=*/none,
      /*input_gate_bias=*/bias_array->getResult(0),
      /*forget_gate_bias=*/bias_array->getResult(1),
      /*cell_bias=*/bias_array->getResult(2),
      /*output_gate_bias=*/bias_array->getResult(3),
      /*projection_weights=*/none,
      /*projection_bias=*/none,
      /*input_activation_state=*/output_init_state,
      /*input_cell_state=*/hidden_init_state,
      /*input_layer_norm_coefficients=*/none,
      /*forget_layer_norm_coefficients=*/none,
      /*cell_layer_norm_coefficients=*/none,
      /*output_layer_norm_coefficients=*/none, builder->getStringAttr("TANH"),
      builder->getF32FloatAttr(10.0), builder->getF32FloatAttr(0.0),
      builder->getStringAttr("FULL"));

  builder->create<mlir::ReturnOp>(func_op.getLoc(), lstm.getResult());
  return success();
}

}  // namespace TFL
}  // namespace mlir
