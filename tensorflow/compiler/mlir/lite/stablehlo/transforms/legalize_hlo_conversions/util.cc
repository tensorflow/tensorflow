/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

bool IsIotaAttr(ArrayRef<int64_t> arr, int64_t size) {
  if (arr.size() != size) return false;
  int64_t iota = 0;
  for (auto s : arr) {
    if (s != iota) return false;
    ++iota;
  }
  return true;
}

PermutationAndShape GetPermutationAndTransposedShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter& rewriter) {
  assert(permutation_array.size() == input_type.getRank());
  llvm::SmallVector<int64_t> transposed_shape(permutation_array.size());
  for (int64_t i = 0; i < permutation_array.size(); ++i) {
    transposed_shape[i] = input_type.getDimSize(permutation_array[i]);
  }
  auto transposed_type =
      RankedTensorType::get(transposed_shape, input_type.getElementType());
  DenseIntElementsAttr permutation = DenseIntElementsAttr::get(
      RankedTensorType::get(permutation_array.size(), rewriter.getI64Type()),
      permutation_array);
  return {permutation, transposed_type};
}

Value BuildIntConstOp(ImplicitLocOpBuilder& builder,
                      ConversionPatternRewriter& rewriter, int64_t const_value,
                      Type type) {
  Value result_const =
      builder.create<TF::ConstOp>(rewriter.getIntegerAttr(type, const_value));
  return result_const;
}

Value BuildIntArrayConstOp(ImplicitLocOpBuilder& builder,
                           ConversionPatternRewriter& rewriter,
                           ArrayRef<int64_t> const_value, Type type) {
  DenseIntElementsAttr const_value_raw;
  if (type == rewriter.getI64Type()) {
    const_value_raw = rewriter.getI64TensorAttr(const_value);
  } else {
    // Convert I64 const array to I32.
    llvm::SmallVector<int32_t> const_i32_vec;
    for (auto element : const_value) {
      const_i32_vec.push_back(static_cast<int32_t>(element));
    }
    const_value_raw = rewriter.getI32TensorAttr(const_i32_vec);
  }
  Value result_const = builder.create<TF::ConstOp>(const_value_raw);
  return result_const;
}

llvm::SmallVector<int64_t> GetInversePermutationArray(
    llvm::ArrayRef<int64_t> permutation_array) {
  llvm::SmallVector<int64_t> inverse_permutation_array(
      permutation_array.size());
  const auto permutation_array_size = permutation_array.size();
  for (int64_t i = 0; i < permutation_array_size; ++i) {
    inverse_permutation_array[permutation_array[i]] = i;
  }
  return inverse_permutation_array;
}

DenseIntElementsAttr GetInversePermutation(
    llvm::ArrayRef<int64_t> permutation_array,
    ConversionPatternRewriter& rewriter) {
  SmallVector<int64_t, 4> inverse_permutation_array =
      GetInversePermutationArray(permutation_array);
  return DenseIntElementsAttr::get(
      RankedTensorType::get(inverse_permutation_array.size(),
                            rewriter.getI64Type()),
      inverse_permutation_array);
}

PermutationAndShape GetInversePermutationAndShape(
    llvm::ArrayRef<int64_t> permutation_array, ShapedType input_type,
    ConversionPatternRewriter& rewriter) {
  SmallVector<int64_t, 4> inverse_permutation_array =
      GetInversePermutationArray(permutation_array);
  return GetPermutationAndTransposedShape(inverse_permutation_array, input_type,
                                          rewriter);
}

LogicalResult NormalizeIndexVector(Operation* parent_op, Value& indices,
                                   ShapedType& indices_type,
                                   int64_t index_vector_dim,
                                   ConversionPatternRewriter& rewriter) {
  if (index_vector_dim == indices_type.getRank()) {
    llvm::SmallVector<int64_t, 4> new_start_indices_shape(
        indices_type.getShape().begin(), indices_type.getShape().end());
    new_start_indices_shape.push_back(1);
    indices_type = RankedTensorType::get(new_start_indices_shape,
                                         indices_type.getElementType());
    indices = rewriter.create<mhlo::ReshapeOp>(parent_op->getLoc(),
                                               indices_type, indices);
  } else if (index_vector_dim != indices_type.getRank() - 1) {
    // If index_vector_dim isn't the last dimension in indices then it isn't
    // supported yet.
    // TODO(tberghammer): Transpose indices to support this usecase.
    return rewriter.notifyMatchFailure(
        parent_op,
        "index vector dim isn't the last dimension in start indices");
  }
  return success();
}

// Check if the specified region is a binary reduction function that takes 2
// inputs and returns the second input. Functions like this are used by update
// scatter like ops.
template <>
LogicalResult MatchBinaryReduceFunction<void>(mlir::Region& function) {
  Block& body = function.front();
  if (body.getNumArguments() != 2) return failure();

  mhlo::ReturnOp return_op = dyn_cast<mhlo::ReturnOp>(body.back());
  if (!return_op) return failure();
  if (return_op.getNumOperands() != 1) return failure();
  if (return_op.getOperands().front() != body.getArgument(1)) return failure();
  return success();
}

Value ConversionState::GetOperand() const {
  if (last_tf_op) {
    return last_tf_op->getResult(0);
  }
  return hlo_op->getOperand(0);
}

TensorType ConversionState::GetOperandTensorType() const {
  if (last_tf_op) {
    return last_tf_op->getResult(0).getType().cast<TensorType>();
  }
  return hlo_op->getOperand(0).getType().cast<TensorType>();
}

llvm::ArrayRef<int64_t> ConversionState::GetOperandShape() const {
  return GetOperandTensorType().getShape();
}

namespace {

// Gets the dilation data for TFLite Dilate.
//
// Depending on the definition of the op we are trying to legalize, a dilation
// can be either seen as interior padding or as a scaling factor where:
//
//     scaling_factor = interior_padding + 1
//
// The is_padding parameter is used to take this difference into account.
llvm::SmallVector<int32_t, 6> GetDilateData(const DenseElementsAttr& dilation,
                                            const bool is_padding) {
  llvm::SmallVector<int32_t, 6> data;
  for (const auto& v : dilation.getValues<APInt>()) {
    data.push_back(v.getSExtValue() + static_cast<int32_t>(is_padding));
  }
  return data;
}

}  // namespace

void AddDilateOpIfRequired(ConversionState& state,
                           const DenseElementsAttr& dilation,
                           const Value padding_value, const bool is_padding) {
  const auto dilate_data = GetDilateData(dilation, is_padding);
  if (absl::c_any_of(dilate_data, IsNot(1))) {
    const TensorType output_type = state.ComputeResultTensorType(
        [](int i, const auto& shape, const auto& dilate_data) {
          if (shape[i] < 0) {
            return shape[i];
          }
          return shape[i] + (shape[i] - 1) * (dilate_data[i] - 1);
        },
        dilate_data);

    auto dilate_tensor = AddConstantTensor(state, dilate_data);
    auto tfl_dilate = state.rewriter.create<TFL::DilateOp>(
        state.hlo_op->getLoc(), output_type, state.GetOperand(), dilate_tensor,
        padding_value);

    state.last_tf_op = tfl_dilate;
  }
}

namespace {

// Gets the pad data for TFLite PadV2.
//
// StableHLO Pad allows negative values for cropping. This functions replaces
// negative values with 0.
llvm::SmallVector<int64_t, 12> GetPadData(
    const DenseElementsAttr& edge_padding_low,
    const DenseElementsAttr& edge_padding_high) {
  llvm::SmallVector<int64_t, 12> data;
  auto low_values = edge_padding_low.getValues<APInt>();
  auto high_values = edge_padding_high.getValues<APInt>();
  for (int i = 0; i < edge_padding_low.getNumElements(); ++i) {
    const int64_t pad_low = low_values[i].getSExtValue();
    const int64_t pad_high = high_values[i].getSExtValue();
    data.push_back(pad_low < 0 ? 0 : pad_low);
    data.push_back(pad_high < 0 ? 0 : pad_high);
  }
  return data;
}

template <class Container>
void AddPadOpIfRequiredImpl(ConversionState& state, const Container& pad_data,
                            const Value padding_value) {
  if (absl::c_any_of(pad_data, IsNot(0))) {
    const TensorType output_type = state.ComputeResultTensorType(
        [](int i, const auto& shape, const auto& pad) {
          if (shape[i] < 0) {
            return shape[i];
          }
          return shape[i] + pad[2 * i] + pad[2 * i + 1];
        },
        pad_data);

    auto pad_tensor = AddConstantTensor(
        state, pad_data,
        {static_cast<int64_t>(state.GetOperandShape().size()), 2});
    auto tfl_pad = state.rewriter.create<TFL::PadV2Op>(
        state.hlo_op->getLoc(), output_type, state.GetOperand(), pad_tensor,
        padding_value);

    state.last_tf_op = tfl_pad;
  }
}

}  // namespace

void AddPadOpIfRequired(ConversionState& state,
                        const DenseElementsAttr& edge_padding_low,
                        const DenseElementsAttr& edge_padding_high,
                        const Value padding_value) {
  AddPadOpIfRequiredImpl(state, GetPadData(edge_padding_low, edge_padding_high),
                         padding_value);
}

namespace {

// Holds the data needed to generate a TFLite StridedSlice operation.
struct StridedSliceData {
  llvm::SmallVector<int64_t, 6> low;
  llvm::SmallVector<int64_t, 6> high;
  llvm::SmallVector<int64_t, 6> strides;
  int32_t begin_mask = 0;
  int32_t end_mask = 0;

  void resize(const size_t size) {
    low.resize(size);
    high.resize(size);
    strides.resize(size);
  }
};

// Updates the strided slice data with the given values for the `i`th element.
//
// Warning: this expects the data internal buffers to have at least i+1
// elements.
void AppendDataDim(StridedSliceData& data, const int i, const APInt& low,
                   const APInt& high, const APInt& stride) {
  const int64_t pad_low = low.getSExtValue();
  const int64_t pad_high = high.getSExtValue();
  if (pad_low >= 0) {
    data.begin_mask |= 1 << i;
    data.low[i] = 0;
  } else {
    data.low[i] = -pad_low;
  }
  if (pad_high >= 0) {
    data.end_mask |= 1 << i;
    data.high[i] = 0;
  } else {
    data.high[i] = pad_high;
  }
  data.strides[i] = stride.getSExtValue();
}

// Gets the data needed to generate a TFLite StridedSlice operation.
StridedSliceData GetStridedSliceData(const DenseElementsAttr& edge_padding_low,
                                     const DenseElementsAttr& edge_padding_high,
                                     const DenseElementsAttr& strides) {
  StridedSliceData data;
  data.resize(edge_padding_low.getNumElements());
  const auto low_values = edge_padding_low.getValues<APInt>();
  const auto high_values = edge_padding_high.getValues<APInt>();
  const auto stride_values = strides.getValues<APInt>();
  for (int i = 0; i < edge_padding_low.getNumElements(); ++i) {
    AppendDataDim(data, i, low_values[i], high_values[i], stride_values[i]);
  }
  return data;
}

void AddStridedSliceOpIfRequiredImpl(
    ConversionState& state, const StridedSliceData& strided_slice_data) {
  if (absl::c_any_of(strided_slice_data.low, IsNot(0)) ||
      absl::c_any_of(strided_slice_data.high, IsNot(0)) ||
      absl::c_any_of(strided_slice_data.strides, IsNot(1))) {
    const TensorType output_type = state.ComputeResultTensorType(
        [](int i, const auto& shape, const auto& high, const auto& low,
           const auto& strides) {
          if (shape[i] < 0) {
            return shape[i];
          }
          return (shape[i] + high[i] - low[i]) / strides[i];
        },
        strided_slice_data.high, strided_slice_data.low,
        strided_slice_data.strides);

    auto crop_begin_tensor = AddConstantTensor(state, strided_slice_data.low);
    auto crop_end_tensor = AddConstantTensor(state, strided_slice_data.high);
    auto crop_strides_tensor =
        AddConstantTensor(state, strided_slice_data.strides);
    auto tfl_crop = state.rewriter.create<TFL::StridedSliceOp>(
        state.hlo_op->getLoc(), output_type, state.GetOperand(),
        crop_begin_tensor, crop_end_tensor, crop_strides_tensor,
        strided_slice_data.begin_mask, strided_slice_data.end_mask, 0, 0, 0,
        false);

    state.last_tf_op = tfl_crop;
  }
}

}  // namespace

bool NeedsReformatTypeAndPermutation(int batch_dim, int feature_dim,
                                     int spatial_dim_start,
                                     int default_batch_dim,
                                     int default_feature_dim,
                                     int default_spatial_dim_start) {
  return batch_dim != default_batch_dim || feature_dim != default_feature_dim ||
         spatial_dim_start != default_spatial_dim_start;
}

std::pair<RankedTensorType, DenseIntElementsAttr> GetReformatTypeAndPermutation(
    int batch_dim, int feature_dim, int spatial_dim_start,
    int default_batch_dim, int default_feature_dim,
    int default_spatial_dim_start, int num_spatial_dims, RankedTensorType type,
    ConversionPatternRewriter& rewriter) {
  auto shape = type.getShape();
  llvm::SmallVector<int64_t, 4> permutation_array(num_spatial_dims + 2);
  permutation_array[default_batch_dim] = batch_dim;
  permutation_array[default_feature_dim] = feature_dim;
  llvm::SmallVector<int64_t, 4> transposed_shape(num_spatial_dims + 2);
  transposed_shape[default_batch_dim] = shape[batch_dim];
  transposed_shape[default_feature_dim] = shape[feature_dim];
  for (int i : llvm::seq<int>(0, num_spatial_dims)) {
    permutation_array[default_spatial_dim_start + i] = spatial_dim_start + i;
    transposed_shape[default_spatial_dim_start + i] =
        shape[spatial_dim_start + i];
  }
  auto new_type =
      RankedTensorType::get(transposed_shape, type.getElementType());
  auto permutation = DenseIntElementsAttr::get(
      RankedTensorType::get({type.getRank()}, rewriter.getI64Type()),
      permutation_array);
  return {new_type, permutation};
}

Value InsertTranspose(Value value, int batch_dim, int feature_dim,
                      ArrayRef<int64_t> spatial_dimensions,
                      int default_batch_dim, int default_feature_dim,
                      int default_spatial_dim_start, int num_spatial_dims,
                      ConversionPatternRewriter& rewriter) {
  auto type = value.getType().cast<RankedTensorType>();
  DenseIntElementsAttr permutation;
  const int spatial_dim_start = spatial_dimensions.front();
  if (!NeedsReformatTypeAndPermutation(
          batch_dim, feature_dim, spatial_dim_start, default_batch_dim,
          default_feature_dim, default_spatial_dim_start)) {
    // Transpose is not needed because the current format is the same a default
    // format.
    return value;
  }
  std::pair<RankedTensorType&, DenseIntElementsAttr&>(type, permutation) =
      GetReformatTypeAndPermutation(batch_dim, feature_dim, spatial_dim_start,
                                    default_batch_dim, default_feature_dim,
                                    default_spatial_dim_start, num_spatial_dims,
                                    type, rewriter);
  return rewriter.create<mhlo::TransposeOp>(value.getLoc(), type, value,
                                            permutation);
}

void AddStridedSliceOpIfRequired(ConversionState& state,
                                 const DenseElementsAttr& edge_padding_low,
                                 const DenseElementsAttr& edge_padding_high,
                                 const DenseElementsAttr& strides) {
  StridedSliceData strided_slice_data =
      GetStridedSliceData(edge_padding_low, edge_padding_high, strides);
  AddStridedSliceOpIfRequiredImpl(state, strided_slice_data);
}

Value CreateCastToInt32(Value val, Location loc, PatternRewriter& rewriter) {
  IntegerType new_ele_type = rewriter.getIntegerType(32);
  if (auto shaped_type = val.getType().dyn_cast<RankedTensorType>()) {
    ShapedType new_type =
        RankedTensorType::get(shaped_type.getShape(), new_ele_type);
    return rewriter.create<TFL::CastOp>(loc, new_type, val);
  }
  return rewriter.create<TFL::CastOp>(
      loc, UnrankedTensorType::get(new_ele_type), val);
}

}  // namespace odml
}  // namespace mlir
