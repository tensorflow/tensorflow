/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass takes operations in TensorFlowLite dialect and
// optimizes them to resulting operations in TensorFlowLite dialect.

#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/optimize_pass_options.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/constant_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {

constexpr char kRelu[] = "RELU";
constexpr char kRelu6[] = "RELU6";
constexpr char kRelu1[] = "RELU_N1_TO_1";

ElementsAttr FlattenTo1D(Attribute a) {
  auto elements = mlir::cast<DenseElementsAttr>(a);
  const std::array<int64_t, 1> flattened_shape = {elements.getNumElements()};
  auto new_type = RankedTensorType::get(flattened_shape,
                                        elements.getType().getElementType());
  return elements.reshape(new_type);
}

// This assumes that the bias is of shape NxCx1x1 and doesn't require transpose
// Its corresponding constraint is optimize_patterns.td:IsBiasShape()
ElementsAttr ReshapeNCHWBiasToNHWC(Value v, Attribute a) {
  auto elements = mlir::cast<DenseElementsAttr>(a);
  auto shape = mlir::cast<ShapedType>(v.getType()).getShape();
  if (shape.size() != 4 || shape[2] != 1 || shape[3] != 1) return elements;
  const std::array<int64_t, 4> new_shape = {shape[0], shape[2], shape[3],
                                            shape[1]};
  auto new_type =
      RankedTensorType::get(new_shape, elements.getType().getElementType());
  return elements.reshape(new_type);
}

bool L2NormalizeReduceAxis(Value sq_op, DenseElementsAttr axis) {
  if (axis.getNumElements() == 0) {
    return false;
  }
  if (mlir::cast<ShapedType>(sq_op.getType()).getRank() - 1 ==
          *axis.getValues<int>().begin() ||
      *axis.getValues<int>().begin() == -1) {
    return true;
  }
  if (mlir::cast<ShapedType>(sq_op.getType()).getRank() !=
      axis.getNumElements()) {
    return false;
  }
  auto shape = mlir::cast<ShapedType>(sq_op.getType());
  SmallVector<int, 4> elems{axis.getValues<int>().begin(),
                            axis.getValues<int>().end()};
  for (int i = 0; i < shape.getRank(); ++i) {
    if (i != elems[i]) return false;
  }
  return true;
}

// Is rankx2xi32 padding array "balanced"
// i.e. 0 <= [d][1] - [d][0] <= 1 for all spatial dims d (and 0 elsewhere).
template <typename T>
bool IsBalancedPaddingArray(int spatials_start, int spatials_end,
                            llvm::ArrayRef<T> data) {
  for (int i = 0; i < data.size() / 2; ++i) {
    const T pad_low = data[2 * i];
    const T pad_hi = data[2 * i + 1];
    if ((i < spatials_start || i >= spatials_end) &&
        (pad_low != 0 || pad_hi != 0)) {
      return false;
    }
    const T pad_diff = pad_hi - pad_low;
    if (pad_diff > 1 || pad_diff < 0) {
      return false;
    }
  }
  return true;
}

bool IsBalancedPaddingArray(int spatials_start, int spatials_end,
                            DenseElementsAttr data) {
  if (data.isSplat()) {
    return false;
  }
  if (data.getElementType().isInteger(64)) {
    return IsBalancedPaddingArray<int64_t>(
        spatials_start, spatials_end,
        llvm::SmallVector<int64_t>(data.value_begin<int64_t>(),
                                   data.value_end<int64_t>()));
  }
  if (data.getElementType().isInteger(32)) {
    return IsBalancedPaddingArray<int32_t>(
        spatials_start, spatials_end,
        llvm::SmallVector<int32_t>(data.value_begin<int32_t>(),
                                   data.value_end<int32_t>()));
  }
  return false;
}

bool HasSameStridedDim(int in, int dilate, int stride, int k, int p) {
  const int effective_filter = (k - 1) * dilate + 1;
  const int out_size = (in + stride - 1) / stride;
  const int padding_needed = (out_size - 1) * stride + effective_filter - in;
  return padding_needed == p;
}

// Is the pre pad shape amenable to given conv with SAME padding.
bool HasSameStridedShape(TFL::Conv2DOp op, ArrayRef<int64_t> pre_pad_shape) {
  auto conv_in_shape =
      llvm::dyn_cast<ShapedType>(op.getInput().getType()).getShape();
  auto kernel_shape =
      llvm::dyn_cast<ShapedType>(op.getFilter().getType()).getShape();
  if (conv_in_shape.size() != kernel_shape.size()) {
    return false;
  }
  if (conv_in_shape.size() < 3) {
    return false;
  }

  const int64_t h_pad = conv_in_shape[1] - pre_pad_shape[1];
  const bool h_strided =
      HasSameStridedDim(pre_pad_shape[1], op.getDilationHFactor(),
                        op.getStrideH(), kernel_shape[1], h_pad);

  const int64_t w_pad = conv_in_shape[2] - pre_pad_shape[2];
  const bool w_strided =
      HasSameStridedDim(pre_pad_shape[2], op.getDilationWFactor(),
                        op.getStrideW(), kernel_shape[2], w_pad);
  return h_strided && w_strided;
}

bool HasSameStridedShape(TFL::DepthwiseConv2DOp op,
                         ArrayRef<int64_t> pre_pad_shape) {
  auto conv_in_shape =
      llvm::dyn_cast<ShapedType>(op.getInput().getType()).getShape();
  auto kernel_shape =
      llvm::dyn_cast<ShapedType>(op.getFilter().getType()).getShape();

  const int64_t h_pad = conv_in_shape[1] - pre_pad_shape[1];
  const bool h_strided =
      HasSameStridedDim(pre_pad_shape[1], op.getDilationHFactor(),
                        op.getStrideH(), kernel_shape[1], h_pad);

  const int64_t w_pad = conv_in_shape[2] - pre_pad_shape[2];
  const bool w_strided =
      HasSameStridedDim(pre_pad_shape[2], op.getDilationWFactor(),
                        op.getStrideW(), kernel_shape[2], w_pad);
  return h_strided && w_strided;
}

bool HasSameStridedShape(TFL::Conv3DOp op, ArrayRef<int64_t> pre_pad_shape) {
  auto conv_in_shape =
      llvm::dyn_cast<ShapedType>(op.getInput().getType()).getShape();
  auto kernel_shape =
      llvm::dyn_cast<ShapedType>(op.getFilter().getType()).getShape();

  const int64_t d_pad = conv_in_shape[1] - pre_pad_shape[1];
  const bool d_strided =
      HasSameStridedDim(pre_pad_shape[1], op.getDilationDFactor(),
                        op.getStrideD(), kernel_shape[0], d_pad);

  const int64_t h_pad = conv_in_shape[2] - pre_pad_shape[2];
  const bool h_strided =
      HasSameStridedDim(pre_pad_shape[2], op.getDilationHFactor(),
                        op.getStrideH(), kernel_shape[1], h_pad);

  const int64_t w_pad = conv_in_shape[3] - pre_pad_shape[3];
  const bool w_strided =
      HasSameStridedDim(pre_pad_shape[3], op.getDilationWFactor(),
                        op.getStrideW(), kernel_shape[2], w_pad);
  return h_strided && w_strided && d_strided;
}

using ::llvm::cast;

// Predicate to check if the product of last few dimensions in LHS is equal to
// the last dimension in RHS.
// agg_start_idx is the index in LHS from where the subsection will start.
bool ContractingDimsProductEqual(Value input, Value output,
                                 size_t agg_start_idx) {
  ArrayRef<int64_t> input_shape =
      mlir::cast<ShapedType>(input.getType()).getShape();
  ArrayRef<int64_t> output_shape =
      mlir::cast<ShapedType>(output.getType()).getShape();

  int agg_value = 1;
  for (size_t i = agg_start_idx; i < input_shape.size(); ++i) {
    agg_value *= input_shape[i];
  }

  return (agg_value == output_shape[output_shape.size() - 1]);
}

// Return true if the product of dimension values of a subsection of the
// tensor is equal to the non-contracting dimension after a reshape
bool NonBroadcastingNonContractingDimsProductEqual(Value original,
                                                   Value updated, bool is_lhs,
                                                   size_t agg_start_idx,
                                                   size_t agg_end_idx = 0) {
  ArrayRef<int64_t> original_shape =
      mlir::cast<ShapedType>(original.getType()).getShape();
  ArrayRef<int64_t> updated_shape =
      mlir::cast<ShapedType>(updated.getType()).getShape();

  int64_t agg_value = 1;
  // If the end_index is not supplied, we'll assume that the contracting
  // dimension count is one and skip the one contracting dimension.
  if (agg_end_idx == 0) {
    if (is_lhs) {
      agg_end_idx = original_shape.size() - 2;
    } else {
      agg_end_idx = original_shape.size() - 1;
    }
  }
  for (size_t i = agg_start_idx; i <= agg_end_idx; ++i) {
    agg_value *= original_shape[i];
  }

  if (is_lhs) {
    return (agg_value == updated_shape[updated_shape.size() - 2]);
  } else {
    return (agg_value == updated_shape[updated_shape.size() - 1]);
  }
}

// Returns whether the given type `a` is broadcast-compatible with `b`.
bool IsBroadcastableElementsAttrAndType(Type a, Type b) {
  return OpTrait::util::getBroadcastedType(a, b) != Type();
}

// Returns whether if `type1` dimensions are the same as the ending dimensions
// of `type2`. This is more restricted than broadcastable.
bool IsTailOfShape(Type type1, Type type2) {
  auto tail_type = mlir::dyn_cast<ShapedType>(type1);
  auto full_type = mlir::dyn_cast<ShapedType>(type2);
  if (!tail_type || !full_type || !tail_type.hasRank() ||
      !full_type.hasRank() || tail_type.getRank() > full_type.getRank())
    return false;
  auto i1 = tail_type.getShape().rbegin(), e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();
  return std::equal(i1, e1, i2);
}

// This function removes explicit broadcasting on type1 and returns whether if
// the reduced `type1` dimensions are the same as the ending dimensions
// of `type2`.
bool IsReducedTailOfShape(Type type1, Type type2) {
  auto tail_type = mlir::dyn_cast<ShapedType>(type1);
  auto full_type = mlir::dyn_cast<ShapedType>(type2);
  if (!tail_type || !full_type || !tail_type.hasRank() || !full_type.hasRank())
    return false;

  auto i1 = tail_type.getShape().rbegin();
  auto reduced_e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();

  while ((std::distance(i1, reduced_e1) > 0) && (*(reduced_e1 - 1) == 1)) {
    reduced_e1--;
  }

  return (std::distance(i1, reduced_e1) > 0) &&
         (std::distance(i1, reduced_e1) <= full_type.getRank()) &&
         (std::equal(i1, reduced_e1, i2));
}

// Check if the value of the last dimension of type1 is equal to the number of
// elements in type2. This is a required condition to flatten type2 to form a
// 1D array and allow the binaryOp handle the broadcasting implicitly.
bool IsLastDimEqualToNumElements(Type type1, Type type2) {
  return (mlir::cast<ShapedType>(type1).getRank() >= 1 &&
          mlir::cast<ShapedType>(type1).getDimSize(
              mlir::cast<ShapedType>(type1).getRank() - 1) ==
              mlir::cast<ShapedType>(type2).getNumElements());
}

bool CanFuseConvOrDepthwiseConvShapes(const ArrayRef<int64_t> filter_shape,
                                      const ArrayRef<int64_t> elements_shape,
                                      bool is_depthwise) {
  // Val tensor must be a scalar or of a shape [1, ... , 1, elements_depth].
  const int elements_rank = elements_shape.size();
  for (int i = 0; i < elements_rank - 1; ++i) {
    if (elements_shape[i] != 1) {
      return false;
    }
  }

  auto elements_depth = elements_shape.empty() ? 1 : elements_shape.back();
  // If elements depth equals 1 (i.e., scalar or tensor with 1 element), then
  // we can let binary op to broadcast elements.
  if (elements_depth == 1) {
    return true;
  }

  // In TFLite Conv2D uses OHWI format for filter, and 1HWO for Depthwise
  // Conv. For conv: Check if last dimension in filter equals the first
  // dimension For depthwise conv: Check if the first in filter dimension
  // equals the first dimension.
  if (filter_shape.empty() ||
      (is_depthwise ? filter_shape.back() != elements_depth
                    : filter_shape[0] != elements_depth))
    return false;
  return true;
}

bool CanFuseConvOrDepthwiseConv(Value filter, Attribute val,
                                bool is_depthwise) {
  const auto elements = mlir::dyn_cast<DenseElementsAttr>(val);
  if (!elements) {
    return false;
  }
  const auto elements_shape = elements.getType().getShape();
  const auto filter_shape = mlir::cast<ShapedType>(filter.getType()).getShape();
  return CanFuseConvOrDepthwiseConvShapes(filter_shape, elements_shape,
                                          is_depthwise);
}

bool CanFuseConvOrDepthwiseConv(Attribute filter, Attribute val,
                                bool is_depthwise) {
  if (const auto elements = mlir::dyn_cast<DenseElementsAttr>(val)) {
    if (const auto filter_elements =
            mlir::dyn_cast<DenseElementsAttr>(filter)) {
      return CanFuseConvOrDepthwiseConvShapes(
          filter_elements.getType().getShape(), elements.getType().getShape(),
          is_depthwise);
    }
  }
  return false;
}

// Returns true if we can eliminate the GatherNdOp or ScatterNdOp. When the
// value of `indices` are from 0 to n-1, the output tensor are identical to
// the `params`.
bool CanOptimizeIdentityGatherNdOrScatterNdOp(Value params,
                                              DenseIntElementsAttr indices,
                                              Type output_type) {
  auto params_type = mlir::dyn_cast<RankedTensorType>(params.getType());
  auto indices_type = mlir::dyn_cast<RankedTensorType>(indices.getType());
  // Checks the shape of `params` is [n, ...], shape of `indices` is [n, 1].
  // 2D `indices` means it gets the first row of `params`. As long as indices
  // iterate the first row of `params`, the output is identical to input.
  if (!params_type || !indices_type || indices_type.getRank() != 2 ||
      indices_type.getDimSize(0) != params_type.getDimSize(0) ||
      indices_type.getDimSize(1) != 1)
    return false;

  // Checks the `params_type` is equal to `output_type`. If not equal, we
  // cannot replace the scatter_nd/gather_nd op with `params`.
  if (params_type != output_type) return false;

  // Checks the value in `indices` is from 0 to n-1.
  int cur_value = 0;
  for (const auto &v : indices.getValues<APInt>()) {
    if (v.getSExtValue() != cur_value) return false;
    ++cur_value;
  }

  return true;
}

// Returns true if we can eliminate the SliceOp. When the values of `begin`
// are all 0s and `size[i]` is equal to either -1 or `input.shape[i]` for each
// dim i, the output tensor is identical to `input`.
bool CanOptimizeIdentitySliceOp(Value input, Attribute begin, Attribute size) {
  // Checks if `begin` and `size` are i32 or i64.
  auto begin_attr = mlir::dyn_cast<DenseIntElementsAttr>(begin);
  auto size_attr = mlir::dyn_cast<DenseIntElementsAttr>(size);
  if (!begin_attr || !size_attr) {
    return false;
  }

  auto begin_elem_ty = begin_attr.getType().getElementType();
  if (!begin_elem_ty.isInteger(32) && !begin_elem_ty.isInteger(64)) {
    return false;
  }
  auto size_elem_ty = size_attr.getType().getElementType();
  if (!size_elem_ty.isInteger(32) && !size_elem_ty.isInteger(64)) {
    return false;
  }

  // Checks if `input` is ranked and its rank is equal to number of elements
  // in `begin` and `size`.
  auto input_ty = mlir::cast<ShapedType>(input.getType());
  if (!input_ty.hasRank()) {
    return false;
  }

  int64_t rank = input_ty.getRank();
  if (rank != begin_attr.getNumElements() ||
      rank != size_attr.getNumElements()) {
    return false;
  }

  // Checks if `begin` is all 0s, and `size[i]` is equal to either -1 or
  // `input.shape[i]`.
  for (uint64_t i = 0; i < rank; ++i) {
    if (begin_attr.getValues<APInt>()[i].getSExtValue() != 0) return false;
    int64_t si = size_attr.getValues<APInt>()[i].getSExtValue();
    if (si != -1 && si != input_ty.getDimSize(i)) return false;
  }

  return true;
}

// Returns 1.0 or -1.0 if binary_op is AddOp or SubOp respectively. Also sets
// the element type of the returned constant to the same of the `base` argument.
// This is used when fusing an Add or a Sub into the bias parameter of a
// convolution.
Value GetBiasMultiplier(OpBuilder &builder, Value binary_op,
                        DenseFPElementsAttr base) {
  ShapedType shaped_type =
      RankedTensorType::get({}, base.getType().getElementType());

  float multiplier =
      (llvm::isa<mlir::TFL::AddOp>(binary_op.getDefiningOp()) ? 1.0 : -1.0);

  auto constant_attr = DenseFPElementsAttr::get(shaped_type, multiplier);
  return builder.create<arith::ConstantOp>(binary_op.getLoc(), constant_attr);
}

bool HasOneTailUnitDimension(Attribute attr) {
  auto elements = mlir::dyn_cast<DenseElementsAttr>(attr);
  if (!elements) {
    return false;
  }
  auto shape = elements.getType().getShape();
  if (shape.empty()) {
    return true;
  }
  // Checks that elements are essentially 1d.
  return elements.getNumElements() == shape.back();
}

// Expand a given attribute to 4D with all 1s except 1 dimension.
// The position of the non-1 dimension is specified by `output_channel_dim`.
// Example: [1, 1, 5], 2 -> [1, 1, 5, 1]
DenseElementsAttr ExpandTo4DForConvImpl(Attribute attr,
                                        int output_channel_dim) {
  assert(HasOneTailUnitDimension(attr));
  auto elements = mlir::dyn_cast<DenseElementsAttr>(attr);
  std::vector<int64_t> shape_data = {1, 1, 1, 1};
  const int vector_length = elements.getNumElements();
  shape_data[output_channel_dim] = vector_length;
  auto new_shape =
      RankedTensorType::get(shape_data, elements.getType().getElementType());
  return elements.reshape(new_shape);
}

DenseElementsAttr ExpandTo4DForConv(Attribute attr,
                                    int output_channel_dim = 0) {
  return ExpandTo4DForConvImpl(attr, output_channel_dim);
}

DenseElementsAttr ExpandTo4DForDepthwiseConv(Attribute a) {
  return ExpandTo4DForConvImpl(a, 3);
}

TypeAttr RescaleQtype(Type input, Attribute factor) {
  return RescaleQuantizedType(input, factor);
}

// Returns `true` if reducing `axes` in `input` with `keep_dims=true` results
// in the specified `shape` and `false` otherwise.
static bool ShapeMatchesReduceWithKeepAxes(Value input,
                                           const mlir::Attribute &axes,
                                           const mlir::Attribute &shape) {
  RankedTensorType type =
      mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!type) return false;

  DenseIntElementsAttr axes_attr =
      mlir::dyn_cast_or_null<DenseIntElementsAttr>(axes);
  DenseIntElementsAttr shape_attr =
      mlir::dyn_cast_or_null<DenseIntElementsAttr>(shape);
  if (!axes_attr || !shape_attr) return false;

  if (shape_attr.getNumElements() != type.getRank()) return false;

  llvm::SmallSet<uint64_t, 4> axes_set;
  for (auto a : axes_attr.getValues<APInt>()) {
    axes_set.insert(a.getZExtValue());
  }

  auto type_shape = type.getShape();
  for (uint64_t i = 0; i < type.getRank(); ++i) {
    if (axes_set.contains(i)) {
      if (shape_attr.getValues<APInt>()[i] != 1) return false;
    } else {
      if (shape_attr.getValues<APInt>()[i] != type_shape[i]) return false;
    }
  }
  return true;
}

// Returns `true` if all the `axes` dimensions of `input` are 1.
static bool AreInputDimensionsOneInAxes(Value input,
                                        const mlir::Attribute &axes) {
  RankedTensorType input_type =
      mlir::dyn_cast_or_null<RankedTensorType>(input.getType());
  if (!input_type) return false;
  auto type_shape = input_type.getShape();

  DenseIntElementsAttr axes_attr =
      mlir::dyn_cast_or_null<DenseIntElementsAttr>(axes);
  if (!axes_attr) return false;

  for (auto a : axes_attr.getValues<APInt>()) {
    int64_t axis = a.getSExtValue();
    if (axis < 0) {
      axis += type_shape.size();
    }
    if (axis < 0 || axis >= type_shape.size()) {
      // `axis` is not a valid axis in input.
      return false;
    }
    if (type_shape[axis] != 1) {
      return false;
    }
  }

  return true;
}

static bool FloatValueEquals(const Attribute &attr, double value) {
  auto fp_attr = mlir::dyn_cast_or_null<DenseFPElementsAttr>(attr);
  if (!fp_attr) return false;

  if (fp_attr.isSplat()) {
    return fp_attr.getSplatValue<APFloat>().isExactlyValue(value);
  }
  return llvm::all_of(fp_attr.getValues<APFloat>(), [value](const APFloat &f) {
    return f.isExactlyValue(value);
  });
}

// Returns true if `value` is compile-time constant and its splat value equals
// to `raw_value`.
template <typename T>
bool IsConstantValueOf(mlir::TypedAttr value, T raw_value) {
  auto element_type = mlir::cast<ShapedType>(value.getType()).getElementType();

  if (mlir::isa<FloatType>(element_type)) {
    return FloatValueEquals(value, raw_value);
  } else if (mlir::isa<IntegerType>(element_type)) {
    auto int_attr = mlir::dyn_cast_or_null<DenseIntElementsAttr>(value);
    if (!int_attr) return false;

    if (int_attr.isSplat()) {
      return int_attr.getSplatValue<APInt>() == raw_value;
    }
    return llvm::all_of(int_attr.getValues<APInt>(),
                        [raw_value](const APInt &f) { return f == raw_value; });
  }

  return false;
}

// Returns true if the value's element type is F32.
bool IsF32Value(Value value) {
  return mlir::cast<ShapedType>(value.getType()).getElementType().isF32();
}

// Returns the number of elements in attr if it is a static shape, 1
// otherwise, as an unranked int32 Attribute.
TypedAttr GetNumElementsOrOne(Type type) {
  auto shaped_type = mlir::cast<ShapedType>(type);
  int32_t num_elements =
      shaped_type.hasStaticShape() ? shaped_type.getNumElements() : 1;

  OpBuilder builder(type.getContext());

  return DenseIntElementsAttr::get(
      RankedTensorType::get({}, builder.getI32Type()),
      {llvm::APInt(32, num_elements, true)});
}

// Reshapes value to a given shape.
Value ReshapeValueDroppingLastDim(OpBuilder &builder, Value value) {
  // This function is always guarded with
  // HasTrivialShapeExceptSecondLastDim(), so we could cast safely here.
  auto type = mlir::cast<ShapedType>(value.getType());
  SmallVector<int> new_shape;
  if (type.hasStaticShape()) {
    for (int64_t dim : type.getShape().drop_back()) {
      new_shape.push_back(dim);
    }
  } else {
    new_shape.push_back(-1);
  }
  return builder.create<ReshapeOp>(
      value.getLoc(), value,
      builder.create<arith::ConstantOp>(
          value.getLoc(),
          DenseIntElementsAttr::get(
              RankedTensorType::get(type.getRank() - 1, builder.getI32Type()),
              new_shape)));
}

// Returns true if val has a static shape and the last dimension equals 1.
bool IsLastDimensionEqualOne(Value val) {
  const auto val_type = mlir::cast<ShapedType>(val.getType());
  if (!val_type.hasStaticShape()) return false;
  const auto val_shape = val_type.getShape();
  if (val_shape.empty()) return false;
  const auto last_element = *val_shape.rbegin();
  return last_element == 1;
}

// Returns true if the supplied value-
// 1) Has only one use or
// 2) Is only used by binary op like AddOp, SubOp, MulOp or DivOp.
bool HasOneUseOrUsedByOnlyBinaryOps(Value out_value) {
  if (out_value.hasOneUse()) {
    return true;
  }

  for (auto &use : out_value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (!llvm::isa<mlir::TFL::AddOp>(owner) &&
        !llvm::isa<mlir::TFL::SubOp>(owner) &&
        !llvm::isa<mlir::TFL::DivOp>(owner) &&
        !llvm::isa<mlir::TFL::MulOp>(owner)) {
      return false;
    }
  }

  return true;
}

// Returns true if attr is a DenseIntElementsAttr of int32 or int64 values or
// an incrementing sequence from 0 to N-1.
//
// If such a value is used in an Equal operator, it can be replaced with
// OneHot.
bool IsOneHotIndexAttribute(Attribute attr) {
  const auto dense_attr = mlir::dyn_cast_or_null<DenseIntElementsAttr>(attr);
  if (!dense_attr) {
    return false;
  }
  auto index_type = dense_attr.getType();
  const auto index_elem_bits = index_type.getElementTypeBitWidth();
  if (index_elem_bits != 32 && index_elem_bits != 64) {
    return false;
  }
  // Checks that the index has shape of [1, 1, 1, ..., 1, N].
  if (index_type.getRank() < 1 ||
      llvm::any_of(index_type.getShape().drop_back(),
                   [](int64_t dim) { return dim != 1; })) {
    return false;
  }
  const auto elems = dense_attr.value_begin<APInt>();
  for (int i = 0; i < dense_attr.getNumElements(); ++i) {
    if (i != elems[i]) {
      return false;
    }
  }
  return true;
}

Value Get1DShapeValue(OpBuilder &builder, Value value) {
  auto type = mlir::cast<ShapedType>(value.getType());
  if (!type.hasStaticShape()) {
    return nullptr;
  }
  auto output_type = RankedTensorType::get({1}, builder.getI32Type());
  const int num_elements = type.getNumElements();
  return builder.create<ConstOp>(
      value.getLoc(), output_type,
      DenseIntElementsAttr::get(output_type, num_elements));
}

Type GetEmbeddingLookupShape(Value lookup, Value value) {
  auto lookup_type = mlir::cast<ShapedType>(lookup.getType());
  if (!lookup_type.hasStaticShape()) {
    return nullptr;
  }
  auto value_type = mlir::cast<ShapedType>(value.getType());
  if (!value_type.hasStaticShape() || value_type.getRank() != 2) {
    return nullptr;
  }
  SmallVector<int64_t> new_shape = {lookup_type.getNumElements(),
                                    value_type.getDimSize(0)};
  return value_type.clone(new_shape);
}

// Creates FullyConnected op from params and returns the output.
mlir::Value GetFcOutput(OpBuilder *builder,
                        ::mlir::Operation::result_range result, Value input,
                        Value filter, Value bias,
                        StringAttr fused_activation_function,
                        StringAttr weights_format, BoolAttr keep_num_dims,
                        BoolAttr asymmetric_quantize_inputs) {
  auto fc_op = builder->create<FullyConnectedOp>(
      result[0].getLoc(), result.getTypes(), input, filter, bias,
      fused_activation_function, weights_format, keep_num_dims,
      asymmetric_quantize_inputs);
  return fc_op->getResult(0);
}

// Returns true if 'value' represents a const ElementsAttr with all values
// equals to 0.0.
bool AllValuesAreZero(mlir::Value value) {
  if (!value) return false;
  DenseElementsAttr vals;
  if (!matchPattern(value, m_Constant(&vals))) return false;
  for (auto elem : vals.getValues<float>())
    if (elem != 0.0f) return false;
  return true;
}

bool IsF32Splat(Attribute input_splat) {
  if (!input_splat) return false;
  auto val = dyn_cast_or_null<DenseElementsAttr>(input_splat);
  if (val) {
    return !val.empty() && val.getElementType().isF32() && val.isSplat();
  }
  return false;
}

// Converts an Attribute with a single value of float or integral type to an
// Attribute holding a single value of float type. If attr has no elements,
// the result is 0.0f.
TypedAttr ConvertSingleElementAttrToFloatAttr(Attribute attr) {
  const auto dense_fp_attr = mlir::dyn_cast_or_null<DenseFPElementsAttr>(attr);
  if (dense_fp_attr) {
    // Already float => return
    return dense_fp_attr;
  }

  OpBuilder builder(attr.getContext());

  const auto dense_int_attr = mlir::dyn_cast<DenseIntElementsAttr>(attr);
  const auto int_values = dense_int_attr.getValues<APInt>();
  float float_val = 0.0f;
  if (!int_values.empty()) {
    const APInt apint_val = *int_values.begin();
    if (dense_int_attr.getType().getElementType().isSignedInteger()) {
      // Get the sign-extended value (=>int64) if the type is signed.
      float_val = apint_val.getSExtValue();
    } else {
      // Get the zero-extended value (=>uint64) if unsigned or signless.
      float_val = apint_val.getZExtValue();
    }
  }
  return DenseFPElementsAttr::get(
      RankedTensorType::get({}, builder.getF32Type()),
      {llvm::APFloat(float_val)});
}

bool IsPermutationNCHW(Value perm) {
  DenseIntElementsAttr perm_const;
  if (!matchPattern(perm, m_Constant(&perm_const))) return false;

  SmallVector<int64_t, 4> axes;
  for (const auto &axis_int : perm_const.getValues<APInt>()) {
    axes.push_back(axis_int.getSExtValue());
  }

  return (axes == SmallVector<int64_t>({0, 3, 1, 2}));
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize.inc"

// Get the number of leading 1s in the shape of the given input.
// Ex. input_shape = [1 x 1 x 1 x 1 x 2 x 1] => 4
// returns 0 if the input shape is not static.
int GetNumLeadingOnes(ShapedType input_type) {
  if (!input_type.hasStaticShape()) return 0;
  auto input_shape = input_type.getShape();
  int num_leading_broadcast_dims = 0;
  for (int i = 0; i < input_shape.size(); ++i) {
    if (input_shape[i] == 1) {
      ++num_leading_broadcast_dims;
    } else {
      break;
    }
  }
  return num_leading_broadcast_dims;
}

// Return the number of trailing 1s in the shape of the given input.
// Ex. input_shape = [1 x 1 x 2 x 1] => 1
// returns 0 if the input shape is not static.
int GetNumTrailingOnes(ShapedType input_type) {
  if (!input_type.hasStaticShape()) return 0;
  auto input_shape = input_type.getShape();
  int num_trailing_broadcast_dims = 0;
  for (int i = input_shape.size() - 1; i >= 0; --i) {
    if (input_shape[i] == 1) {
      ++num_trailing_broadcast_dims;
    } else {
      break;
    }
  }
  return num_trailing_broadcast_dims;
}

// Consider as Reshape(
//               Broadcast(
//                 Reshape(input, // input_shape=[1 x n]
//                         inner_shape), // inner_shape=[1 x 1 x 1 x n x 1 x 1]
//                 broadcast_shape), // broadcast_shape=[1 x 1 x 1 x n x m x 1]
//               outer_shape))) // outer_shape=[1 x 1 x n*m]
// Here the broadcast operation is used to create `m` repetetions of the `n`
// elements in the origiginal tensor, making a total of `m*n` number of elements
// in the final tensor that will then be reshaped to form something like
// [1 x 1 x 1 x m*n] by the outermost reshape_op.
// problem: The inefficiency here is that the innermost reshape_op and the
// broadcast_op are introducing unnecessary leading and trailing 1s'.
// fix: Remove the unnecessary 1s' in the inner reshape_op and broadcast_op.
struct SqueezeReshapesAroundBroadcastOp
    : public OpRewritePattern<TFL::BroadcastToOp> {
  using OpRewritePattern<TFL::BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::BroadcastToOp tfl_broadcast_to_op,
                                PatternRewriter &rewriter) const override {
    auto loc = tfl_broadcast_to_op->getLoc();

    // Match the
    // Reshape(
    //   Broadcast(
    //     Reshape(input,inner_shape),
    //     broadcast_shape),
    //   outer_shape))) pattern.
    if (!llvm::dyn_cast_or_null<TFL::ReshapeOp>(
            tfl_broadcast_to_op.getInput().getDefiningOp()) ||
        // Check that the broadcast_to op has only one use.
        !tfl_broadcast_to_op.getOutput().hasOneUse() ||
        !llvm::dyn_cast_or_null<TFL::ReshapeOp>(
            *tfl_broadcast_to_op.getOutput().getUsers().begin())) {
      return rewriter.notifyMatchFailure(
          loc, "No Reshape->BroadcastTo->Reshape pattern found");
    }

    // Pattern is applied only if the broadcast_to shape has more than 5
    // dimensions.
    if (mlir::cast<ShapedType>(tfl_broadcast_to_op.getShape().getType())
            .getNumElements() < 6) {
      return rewriter.notifyMatchFailure(loc,
                                         "Not supported broadcast_to shape");
    }
    auto inner_reshape_op = llvm::dyn_cast_or_null<TFL::ReshapeOp>(
        tfl_broadcast_to_op.getInput().getDefiningOp());
    auto inner_reshape_input = inner_reshape_op.getInput();
    auto outer_reshape_op = llvm::dyn_cast_or_null<TFL::ReshapeOp>(
        *tfl_broadcast_to_op.getOutput().getUsers().begin());

    // Check that the outermost reshape_op in the pattern does not add
    // additional elements to the final output tensor.
    // TODO: b/323217483. This code needs to generalized to additional cases.
    // For example- inner-shape = [1, 1, 1, 8, 1, 10],
    // broadcast_shape = [1, 1, 1, 8, 16, 10] & outer_shape = [1, 1, 1, 1280, 1]
    // And extend the pettern to handle dynamic shapes.
    if (!inner_reshape_op.getOutput().getType().hasStaticShape() ||
        !tfl_broadcast_to_op.getOutput().getType().hasStaticShape() ||
        !outer_reshape_op.getOutput().getType().hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          loc, "Unsupported shapes. Currely only static shapes are supported");
    }

    if (!IsLastDimEqualToNumElements(inner_reshape_input.getType(),
                                     inner_reshape_op.getOutput().getType()) ||
        !IsLastDimEqualToNumElements(
            outer_reshape_op.getOutput().getType(),
            tfl_broadcast_to_op.getOutput().getType())) {
      return rewriter.notifyMatchFailure(
          loc, "Not supported Reshape->BroadcastTo->Reshape pattern");
    }

    // Calculate the number of extra leading and trailing 1s in the
    // broadcast_op output.
    auto broadcast_output_shapetype =
        mlir::cast<ShapedType>(tfl_broadcast_to_op.getOutput().getType());
    int num_leading_broadcast_dims =
        GetNumLeadingOnes(broadcast_output_shapetype);
    int num_trailing_broadcast_dims =
        GetNumTrailingOnes(broadcast_output_shapetype);

    // Get the new shape for the inner reshape_op after removing the extra 1s.
    llvm::SmallVector<int32_t, 6> new_reshape_shape_i32{
        mlir::cast<RankedTensorType>(inner_reshape_op.getOutput().getType())
            .getShape()
            .drop_back(num_trailing_broadcast_dims)
            .drop_front(num_leading_broadcast_dims)};

    Value new_reshape_shape_value = rewriter.create<arith::ConstantOp>(
        inner_reshape_op->getLoc(),
        GetI32ElementsAttr(new_reshape_shape_i32, &rewriter));

    auto new_inner_reshape_op = rewriter.create<TFL::ReshapeOp>(
        inner_reshape_op->getLoc(), inner_reshape_input,
        new_reshape_shape_value);

    // Create a new reshape_op to replace the old inner reshape_op.
    rewriter.replaceOp(inner_reshape_op, new_inner_reshape_op.getResult());

    // Get the new shape for the broadcast_op after removing the extra 1s.
    llvm::SmallVector<int64_t, 6> new_broadcast_shape{
        broadcast_output_shapetype.getShape()
            .drop_back(num_trailing_broadcast_dims)
            .drop_front(num_leading_broadcast_dims)};

    Value new_broadcast_shape_value = rewriter.create<arith::ConstantOp>(
        loc, GetI64ElementsAttr(new_broadcast_shape, &rewriter));

    auto new_broadcast_to_op = rewriter.create<TFL::BroadcastToOp>(
        loc, RankedTensorType::get(new_broadcast_shape, rewriter.getF32Type()),
        new_inner_reshape_op.getOutput(), new_broadcast_shape_value);

    // Create a new broadcast_op to replace the old broadcast_op.
    rewriter.replaceOp(tfl_broadcast_to_op, new_broadcast_to_op.getResult());

    return success();
  }
};

struct FuseAddAndStridedSlice : public OpRewritePattern<TFL::StridedSliceOp> {
  using OpRewritePattern<TFL::StridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::StridedSliceOp strided_slice_op,
                                PatternRewriter &rewriter) const override {
    // Match Add
    mlir::TFL::AddOp add_op =
        dyn_cast_or_null<TFL::AddOp>(strided_slice_op.getEnd().getDefiningOp());
    mlir::TFL::SubOp sub_op =
        dyn_cast_or_null<TFL::SubOp>(strided_slice_op.getEnd().getDefiningOp());
    if (!(add_op || sub_op)) {
      return failure();
    }

    // Check that add rhs is constant.
    DenseElementsAttr added_value;
    Value constant_val = add_op ? add_op.getRhs() : sub_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&added_value))) return failure();

    // Check the add op is applied to begin.
    mlir::TypedValue<::mlir::TensorType> begin_tensor =
        strided_slice_op.getBegin();
    mlir::TypedValue<::mlir::TensorType> add_source_tensor =
        add_op ? add_op.getLhs() : sub_op.getLhs();
    if (begin_tensor != add_source_tensor) {
      return failure();
    }

    // Check that strides are constant
    DenseElementsAttr strides_value;
    Value strides_val = strided_slice_op.getStrides();
    if (!matchPattern(strides_val, m_Constant(&strides_value)))
      return failure();

    mlir::TensorType constant_val_type =
        mlir::cast<TensorType>(constant_val.getType());
    // If it's not 1D or 0D (which can be broadcasted to 1D), reject the
    // matching.
    if (constant_val_type.getRank() > 1) {
      return failure();
    }

    mlir::RankedTensorType end_type =
        mlir::dyn_cast<RankedTensorType>(strided_slice_op.getEnd().getType());
    // begin, end and strides are Rank 1 tensors with one element per dimension
    // of input.
    int64_t num_dims = end_type.getShape()[0];
    DenseElementsAttr new_added_value =
        added_value.reshape(RankedTensorType::get(
            {num_dims},
            mlir::cast<ShapedType>(added_value.getType()).getElementType()));
    ::mlir::arith::ConstantOp new_end = rewriter.create<arith::ConstantOp>(
        strided_slice_op.getEnd().getLoc(), new_added_value);

    if (strided_slice_op.getBeginMask() != 0) return failure();
    if (strided_slice_op.getEndMask() != 0) return failure();
    if (strided_slice_op.getEllipsisMask() != 0) return failure();
    mlir::TFL::StridedSliceOp new_strided_slice_op =
        rewriter.create<TFL::StridedSliceOp>(
            strided_slice_op.getLoc(), strided_slice_op.getOutput().getType(),
            strided_slice_op.getInput(), strided_slice_op.getBegin(), new_end,
            strided_slice_op.getStrides(), strided_slice_op.getBeginMask(),
            strided_slice_op.getEndMask(), strided_slice_op.getEllipsisMask(),
            strided_slice_op.getNewAxisMask(),
            strided_slice_op.getShrinkAxisMask(),
            /*offset=*/true);
    rewriter.replaceOp(strided_slice_op, new_strided_slice_op.getOutput());

    return success();
  }
};

struct Convert2DUpscalingToResizeNearestNeighor
    : public OpRewritePattern<TFL::GatherNdOp> {
  using OpRewritePattern<TFL::GatherNdOp>::OpRewritePattern;

  // Lowers 2D upscaling logic to a single TFL::ResizeNearestNeighor op.
  //
  // To optimize JAX resize implementation, especially for 2D upscaling, this
  // pattern matching logic captures the following pattern to replace with the
  // single TFL::resize_nearest_neighbor op instance as a fast fused op
  // available in TFLite.
  //
  // - tfl.gather_nd -> tfl.transpose -> tfl.gather_nd -> tfl.transpose
  //   where ...
  //     - all tfl.gather_nd op instances take [0, 0, 1, 1, ..., n-1, n-1] as
  //       the indices argument,
  //     - first transpose op takes perm [2, 1, 0, 3], and
  //     - second transpose op take perm [1, 2, 0, 3].
  //
  // Note the current pattern matching logic only handles when width == height.
  LogicalResult matchAndRewrite(TFL::GatherNdOp gather_nd_first,
                                PatternRewriter &rewriter) const override {
    auto result_value = gather_nd_first.getResult();
    auto params_value = gather_nd_first.getParams();
    auto indices_value = gather_nd_first.getIndices();

    auto result_type = dyn_cast<RankedTensorType>(result_value.getType());
    auto params_type = dyn_cast<RankedTensorType>(params_value.getType());
    auto indices_type = dyn_cast<RankedTensorType>(indices_value.getType());

    // Handle static shape cases only.
    if (!result_type || !params_type || !indices_type) return failure();

    // Handle i32 indices and f32 input only.
    if (indices_type.getElementType() != rewriter.getI32Type() ||
        params_type.getElementType() != rewriter.getF32Type()) {
      return failure();
    }

    // The pattern matching allows arbitrary channel dimension but it handles
    // only when height = width.
    if (params_type.getShape().size() != 4 ||
        indices_type.getShape().size() != 2)
      return failure();
    if (params_type.getShape()[1] != 1) return failure();
    if (params_type.getShape()[0] != params_type.getShape()[2])
      return failure();
    if (result_type.getShape()[0] != params_type.getShape()[0] * 2)
      return failure();

    // Make sure that the current pattern contains the following pattern:
    //  - gather_nd->transpose->gather_nd->transpose.
    if (!gather_nd_first->hasOneUse()) return failure();
    auto transpose_first =
        dyn_cast_or_null<TFL::TransposeOp>(*(gather_nd_first->user_begin()));
    if (!transpose_first || !transpose_first->hasOneUse()) return failure();
    auto gather_nd_second =
        dyn_cast_or_null<TFL::GatherNdOp>(*(transpose_first->user_begin()));
    if (!gather_nd_second || !gather_nd_second->hasOneUse()) return failure();
    auto transpose_second =
        dyn_cast_or_null<TFL::TransposeOp>(*(gather_nd_second->user_begin()));
    if (!transpose_second) return failure();

    // Check whether both gather_nd ops implement dimenaion size doubling.
    DenseIntElementsAttr indices;
    if (!matchPattern(gather_nd_first.getIndices(), m_Constant(&indices)))
      return failure();
    int i = 0;
    for (const auto &axis_int : indices.getValues<APInt>()) {
      const int64_t axis = axis_int.getSExtValue();
      if (axis != i / 2) return failure();
      ++i;
    }
    if (!matchPattern(gather_nd_second.getIndices(), m_Constant(&indices)))
      return failure();
    i = 0;
    for (const auto &axis_int : indices.getValues<APInt>()) {
      const int64_t axis = axis_int.getSExtValue();
      if (axis != i / 2) return failure();
      ++i;
    }

    // Check whether first transpose's perm has [2, 1, 0, 3].
    DenseIntElementsAttr perm;
    if (!matchPattern(transpose_first.getPerm(), m_Constant(&perm)))
      return failure();
    SmallVector<int64_t, 4> axes;
    for (const auto &axis_int : perm.getValues<APInt>()) {
      axes.push_back(axis_int.getSExtValue());
    }
    if (axes != SmallVector<int64_t>({2, 1, 0, 3})) return failure();

    // Check whether second transpose's perm has [1, 2, 0, 3].
    if (!matchPattern(transpose_second.getPerm(), m_Constant(&perm)))
      return failure();
    axes.clear();
    for (const auto &axis_int : perm.getValues<APInt>()) {
      axes.push_back(axis_int.getSExtValue());
    }
    if (axes != SmallVector<int64_t>({1, 2, 0, 3})) return failure();

    // Add reshape op to be aligned with the input restriction with
    // TFL::resize_nearest_neighor op.
    const int32_t image_size = static_cast<int32_t>(params_type.getShape()[0]);
    const int32_t feature_size =
        static_cast<int32_t>(params_type.getShape()[3]);
    SmallVector<int32_t, 4> reshape_shape(
        {1, image_size, image_size, feature_size});
    SmallVector<int64_t, 4> reshape_shape_in_int64(
        {1, image_size, image_size, feature_size});

    auto reshape_shape_const_op = rewriter.create<TFL::ConstOp>(
        gather_nd_first->getLoc(),
        GetI32ElementsAttr(reshape_shape, &rewriter));

    auto reshape_op = rewriter.create<TFL::ReshapeOp>(
        gather_nd_first->getLoc(),
        tensorflow::GetTypeFromTFTensorShape(reshape_shape_in_int64,
                                             result_type.getElementType()),
        params_value, reshape_shape_const_op.getResult());

    // Add TFL::resize_nearest_neighor op for 2x upscaling.
    SmallVector<int32_t, 2> size_vec = {image_size * 2, image_size * 2};
    auto size_const_op = rewriter.create<TFL::ConstOp>(
        gather_nd_first->getLoc(), GetI32ElementsAttr(size_vec, &rewriter));

    auto resize = rewriter.create<TFL::ResizeNearestNeighborOp>(
        gather_nd_first->getLoc(), transpose_second.getResult().getType(),
        reshape_op.getResult(), size_const_op.getResult(), false, false);

    rewriter.replaceOp(transpose_second, resize.getResult());
    return success();
  }
};

// Fuse Add with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndAdd : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp add_op,
                                PatternRewriter &rewriter) const override {
    // Match Add.
    DenseElementsAttr added_value;
    Value constant_val = add_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&added_value))) {
      // The constant may be preceded by QDQs in models with QDQ format, so we
      // should set it to the real constant.
      auto dq = dyn_cast_or_null<DequantizeOp>(constant_val.getDefiningOp());
      if (!dq) return failure();
      auto q = dyn_cast_or_null<QuantizeOp>(dq.getInput().getDefiningOp());
      if (!q || !matchPattern(q.getInput(), m_Constant(&added_value))) {
        return failure();
      }
    }

    // Match Fully Connected.
    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        add_op.getLhs().getDefiningOp());
    if (!fc_op) return failure();

    auto constant_val_type = mlir::cast<TensorType>(constant_val.getType());

    // In TFLite FullyConnect definition, bias must be a 1D tensor where
    // the number of elements is equal to the number of channels.
    // If it's not 1D or 0D (which can be broadcasted to 1D), reject the
    // matching.
    bool is_scalar_rhs = false;
    if (constant_val_type.getRank() == 0) {
      is_scalar_rhs = true;
    } else if (constant_val_type.getRank() > 1) {
      return failure();
    }

    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    ElementsAttr bias_value;
    const bool is_none_bias = mlir::isa<NoneType>(bias.getType());
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    if (!is_none_bias && !matchPattern(bias, m_Constant(&bias_value)))
      return failure();

    // Rewrite
    if (is_none_bias) {
      if (is_scalar_rhs) {
        // If the `constant_val` is scalar, we must the shape of filter
        // to properly broadcast the scalar to `{num_channels}` shape.

        // Get the number of channels if possible.
        auto filter_type = mlir::dyn_cast<RankedTensorType>(filter.getType());
        // Filter must be a `2D` tensor with `{num_channels, num_features}`
        // shape. The following check is rejecting unknown rank (-1).
        if (filter_type == nullptr || filter_type.getRank() != 2) {
          return failure();
        }
        int num_channels = filter_type.getShape()[0];

        // Create a zero tensor with shape {num_channels}, and the type need
        // to be the same as constant_val. This is a way to gracefully handle
        // scalar tensor. The Add will always be constant-folded away
        // regardless if `constant_val` is a scalar or not.
        RankedTensorType type = RankedTensorType::get(
            {num_channels}, constant_val_type.getElementType());
        auto attr = rewriter.getZeroAttr(type);
        bias = rewriter.create<arith::ConstantOp>(add_op.getLoc(), type, attr);
        auto none_af = rewriter.getStringAttr("NONE");

        bias =
            rewriter.create<AddOp>(add_op.getLoc(), bias, constant_val, none_af)
                .getOutput();
      } else {
        // If there no pre-existing bias and the `constant_val` is 1D, simply
        // use `constant_val` as bias.
        bias = constant_val;
      }
    } else {
      bias = rewriter
                 .create<AddOp>(add_op.getLoc(), bias, constant_val,
                                rewriter.getStringAttr("NONE"))
                 .getOutput();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), add_op.getLoc()}),
        add_op.getType(),
        /*input=*/fc_op.getInput(),
        /*filter=*/filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(add_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.getWeightsFormat()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/
        fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(add_op, fc.getOutput());

    return success();
  }
};

// Replace ..
// FC(Add(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, filter, FC(rhs, filter, bias))
// .. if rhs, filter, and bias are all constants.
// The second FC will be constant folded to a single vector.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseAddAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
    // This only works with default format.
    if (fc_op.getWeightsFormat() != "DEFAULT") return failure();

    // Match Add.
    auto add_op =
        dyn_cast_or_null<TFL::AddOp>(fc_op.getInput().getDefiningOp());
    if (!add_op) return failure();
    if (add_op.getFusedActivationFunction() != "NONE") return failure();

    // Don't match adds where the added constant is not 1D.
    {
      auto addend_shape = mlir::cast<ShapedType>(add_op.getRhs().getType());
      if (!addend_shape.hasStaticShape()) return failure();
      if (addend_shape.getShape().size() != 1) return failure();
    }

    // Calculate new bias.  Generate a new FC; it will be constant folded.
    auto old_bias = fc_op.getBias();
    if (!old_bias || mlir::isa<NoneType>(old_bias.getType())) {
      // TODO(b/180752069): Figure out new bias' type when old bias is empty.
      return failure();
    }

    // The FC relies on constant folding, which is implemented on F32. Checks
    // types to be F32.
    {
      if (!IsF32Value(add_op.getRhs()) || !IsF32Value(fc_op.getFilter()) ||
          !IsF32Value(old_bias))
        return failure();
    }

    auto new_bias = rewriter.create<TFL::FullyConnectedOp>(
        fc_op.getLoc(), old_bias.getType(),
        /*input=*/add_op.getRhs(),
        /*filter=*/fc_op.getFilter(),
        /*bias=*/old_bias,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(true),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());

    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(add_op.getContext(), {add_op.getLoc(), fc_op.getLoc()}),
        fc_op.getOutput().getTypes(),
        /*input=*/add_op.getLhs(),
        /*filter=*/fc_op.getFilter(),
        /*bias=*/*new_bias.getOutput().begin(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.getOutput());

    return success();
  }
};

// Replace ..
// FC(Mul(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, Mul(filter, rhs), bias)
// .. if rhs, filter, and bias are all constants.
// The generated Mul will be constant folded to a single matrix.
struct FuseMulAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
    // This only works with default format.
    if (fc_op.getWeightsFormat() != "DEFAULT") return failure();

    // Match Mul.
    auto mul_op =
        dyn_cast_or_null<TFL::MulOp>(fc_op.getInput().getDefiningOp());
    if (!mul_op) return failure();
    if (mul_op.getFusedActivationFunction() != "NONE") return failure();

    // Don't match muls where the multiplier constant is not 1D.
    {
      auto multiplier_shape = mlir::cast<ShapedType>(mul_op.getRhs().getType());
      if (!multiplier_shape.hasStaticShape()) return failure();
      if (multiplier_shape.getShape().size() != 1) return failure();
    }

    // We rely on constant folding, implemented only for F32. Check types.
    if (!IsF32Value(mul_op.getRhs()) || !IsF32Value(fc_op.getFilter())) {
      return failure();
    }

    auto location =
        FusedLoc::get(mul_op.getContext(), {mul_op.getLoc(), fc_op.getLoc()});

    auto new_filter = rewriter.create<TFL::MulOp>(
        location,
        /*lhs=*/fc_op.getFilter(),
        /*rhs=*/mul_op.getRhs(),
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"));
    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        location, fc_op.getOutput().getTypes(),
        /*input=*/mul_op.getLhs(),
        /*filter=*/new_filter,
        /*bias=*/fc_op.getBias(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.getOutput());

    return success();
  }
};

// TODO(b/136285429): Move to tablegen when variadic is supported.
template <typename ReluXOp, char const *Act>
struct FuseFullyConnectedAndReluX : public OpRewritePattern<ReluXOp> {
  using OpRewritePattern<ReluXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluXOp relu_op,
                                PatternRewriter &rewriter) const override {
    Operation *input = relu_op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<FullyConnectedOp>(input)) return failure();
    auto fully_connected_op = cast<FullyConnectedOp>(input);
    if (fully_connected_op.getFusedActivationFunction() != "NONE")
      return failure();

    auto new_activation_func = rewriter.getStringAttr(Act);
    auto new_weights_format =
        rewriter.getStringAttr(fully_connected_op.getWeightsFormat());
    auto new_keep_num_dims =
        rewriter.getBoolAttr(fully_connected_op.getKeepNumDims());
    auto fc = rewriter.create<FullyConnectedOp>(
        FusedLoc::get(relu_op.getContext(),
                      {fully_connected_op.getLoc(), relu_op.getLoc()}),
        relu_op.getType(), /*input=*/fully_connected_op.getInput(),
        /*filter=*/fully_connected_op.getFilter(),
        /*bias=*/fully_connected_op.getBias(),
        /*fused_activation_function=*/new_activation_func,
        /*weights_format=*/new_weights_format,
        /*keep_num_dims=*/new_keep_num_dims,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(relu_op, fc.getOutput());

    return success();
  }
};

// Fuse Mul with proceeding FullyConnected.
// Replace ..
// Mul(FC(input, filter, bias), rhs)
// .. with ..
// FC(lhs, Mul(filter, rhs), bias)
// .. if rhs, filter, and bias are all constants.
// The generated Mul will be constant folded to a single matrix using TF::Mul.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndMul : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // If we are broadcasting on the lhs then don't fold the multiply as it
    // would increase the amount of compute done by the fully connected op.
    if (mul_op.getLhs().getType() != mul_op.getType()) return failure();

    // Mul.
    DenseElementsAttr cst;
    Value constant_val = mul_op.getRhs();
    if (!matchPattern(constant_val, m_Constant(&cst))) return failure();

    // Fully Connected.
    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        mul_op.getLhs().getDefiningOp());
    if (!fc_op) return failure();

    // Check if FullyConnected has only one use, that is the LHS of Mul Op.
    // Otherwise this will duplicate the fullyconnected op to serve the
    // remaining uses.
    if (!fc_op->hasOneUse()) return failure();

    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!mlir::isa<NoneType>(bias.getType()) &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.getFusedActivationFunction() != "NONE") return failure();

    // Only fuse multiplier if all dimensions other than the depth dimension
    // are equal to 1 since otherwise
    // `matmul(x, filter) * cst != matmul(x, filter * cst)`
    // even if `filter` and `cst` are be broadcastable.
    auto shape = cst.getType().getShape();
    if (!IsDimensionsDegenerateExceptLastOne(shape)) return failure();

    int64_t element_size = shape.empty() ? 1 : shape[shape.size() - 1];
    // Expand and transpose the multiplier since weights are using the
    // OHWI data format in TFLite.
    int64_t normalized_shape[2] = {element_size, 1};
    auto new_cst = cst.reshape(RankedTensorType::get(
        normalized_shape, cst.getType().getElementType()));
    Type new_type = new_cst.getType();
    if (!IsBroadcastableElementsAttrAndType(new_type, filter.getType())) {
      return failure();
    }

    auto new_op =
        rewriter.create<arith::ConstantOp>(mul_op.getLoc(), new_type, new_cst);
    Value new_const_val = new_op.getResult();

    // Rewrite. Since the folder of TFL::MulOp couldn't broadcast the operands,
    // TF::MulOp is used to fold the constant.
    // TODO(b/139192933): switch to the TFL constant folding
    auto filter_type = mlir::cast<ShapedType>(filter.getType());
    if (filter_type.hasStaticShape()) {
      auto size =
          filter_type.getNumElements() * filter_type.getElementTypeBitWidth();
      // Don't constant fold if the filter is too large for TF to fold.
      // tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.cc
      if (size > (1 << 30)) return failure();
    }
    auto new_filter =
        rewriter.create<TF::MulOp>(mul_op.getLoc(), filter, new_const_val)
            .getZ();
    // If bias isn't None, it needs to be multiplied as well.
    if (!mlir::isa<NoneType>(bias.getType())) {
      bias = rewriter.create<TF::MulOp>(mul_op.getLoc(), bias, constant_val)
                 .getZ();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), mul_op.getLoc()}),
        mul_op.getType(),
        /*input=*/fc_op.getInput(),
        /*filter=*/new_filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(mul_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.getWeightsFormat()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(mul_op, fc.getOutput());

    return success();
  }
};

// Fuse Mul with proceeding Affine ops. This is an C++ implementation of the
// following table gen implementation, which doesn't derived the result type of
// the TFL_DequantizeOp.
// def : Pat<(TFL_MulOp (TFL_Conv2DOp:$conv_output $input,
//                          (TFL_DequantizeOp (TFL_QuantizeOp
//                              (Arith_ConstantOp F32ElementsAttr:$filter),
//                              $qtype)),
//                          (Arith_ConstantOp F32ElementsAttr:$bias),
//                          $h_factor, $w_factor, TFL_AF_None,
//                          $padding, $stride_h, $stride_w),
//                      (Arith_ConstantOp F32ElementsAttr:$value), $act_fn),
//           (TFL_Conv2DOp $input,
//                      (TFL_DequantizeOp (TFL_QuantizeOp
//                          (TFL_MulOp (Arith_ConstantOp $filter),
//                                     (Arith_ConstantOp (ExpandTo4DForConv
//                                     $value)),
//                                      TFL_AF_None),
//                          (RescaleQtype $qtype, $value))),
//                      (TFL_MulOp (Arith_ConstantOp $bias), (Arith_ConstantOp
//                      $value),
//                          TFL_AF_None),
//                      $h_factor, $w_factor, $act_fn,
//                      $padding, $stride_h, $stride_w),
//         [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
//          (HasOneUse $conv_output),
//          (IsPerAxisQuantization $qtype), // per-axis quantization
//         ]>;
template <typename AffineOpType>
struct FuseAffinOpAndMulWithQDQs : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
    // Mul. Required 1-D or squeezable to 1-D rhs for batch normalization.
    DenseElementsAttr gamma_cst;
    Value gamma = mul_op.getRhs();
    if (!matchPattern(gamma, m_Constant(&gamma_cst))) return failure();
    if (!HasOneTailUnitDimension(gamma_cst)) {
      return failure();
    }

    // Affine op
    Operation *mul_op_lhs = mul_op.getLhs().getDefiningOp();
    auto affine_op = dyn_cast_or_null<AffineOpType>(mul_op_lhs);
    if (!affine_op) {
      return failure();
    }
    // This is the tensor feeding the affine op's rhs. This is not necessarily
    // the filter since it could be produced by a transpose or a Q-DQ).
    Value affine_rhs = affine_op.getFilter();
    Value bias = affine_op.getBias();

    // This indicates which dimension of the actual filter constant contains
    // the output channels dimension.
    int filter_output_dim = -1;
    if (isa<TFL::Conv2DOp>(affine_op)) {
      filter_output_dim = 0;
    } else if (isa<TFL::DepthwiseConv2DOp>(affine_op)) {
      filter_output_dim = 3;
    } else {
      return failure();
    }
    // The rhs could be :
    // const -> QDQ -> [transpose | reshape] -> conv
    auto reshape_op =
        dyn_cast_or_null<TFL::ReshapeOp>(affine_rhs.getDefiningOp());
    // If the filter is created by a transpose op, check if the there is QDQ
    // feeding that transpose.
    if (auto transpose_op =
            dyn_cast_or_null<TFL::TransposeOp>(affine_rhs.getDefiningOp())) {
      // If there is a transpose op, reassigning affine_rhs to be the
      // pre-transpose tensor.
      affine_rhs = transpose_op.getInput();
      DenseIntElementsAttr permutation_cst;
      if (!matchPattern(transpose_op.getPerm(), m_Constant(&permutation_cst))) {
        return failure();
      }
      SmallVector<int32_t> permutation_vec =
          llvm::to_vector(permutation_cst.getValues<int32_t>());
      // reversing the effect of the transpose op. For example, if the output
      // dimension is 0, after transpose with a transpose permutation vector of
      // [3, 0, 1, 2], the pre-transpose output dimension (or the actual filter
      // output dimension) would be 3.
      filter_output_dim = permutation_vec[filter_output_dim];
    } else if (reshape_op) {
      affine_rhs = reshape_op.getInput();
    }

    // QDQs
    auto dq_op =
        dyn_cast_or_null<TFL::DequantizeOp>(affine_rhs.getDefiningOp());
    if (!dq_op) {
      return failure();
    }
    auto q_op =
        dyn_cast_or_null<TFL::QuantizeOp>(dq_op.getInput().getDefiningOp());
    if (!q_op) {
      return failure();
    }
    // If the transpose is changed to a reshape op, we'll resort to finding the
    // output channel from the quant_dimension of the Q op if it is per-channel.
    if (reshape_op) {
      auto per_axis_quantized_type =
          dyn_cast<quant::UniformQuantizedPerAxisType>(
              getElementTypeOrSelf(q_op.getType()));
      if (per_axis_quantized_type) {
        filter_output_dim = per_axis_quantized_type.getQuantizedDimension();
      } else {
        // If transpose op is replaced with a reshape op, we've already lost the
        // permutation. If the Q op is not per-channel, we're out of luck to
        // find the channel dimension of the filter.
        return failure();
      }
    }
    Value filter = q_op.getInput();

    // weight constant
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) {
      return failure();
    }
    if (!mlir::isa<NoneType>(bias.getType()) &&
        !matchPattern(bias, m_Constant(&cst_tmp))) {
      return failure();
    }
    if (affine_op.getFusedActivationFunction() != "NONE") {
      return failure();
    }
    rewriter.setInsertionPoint(q_op);
    Location loc = affine_op.getLoc();

    DenseElementsAttr broadcasted_gamma_attr =
        ExpandTo4DForConv(gamma_cst, filter_output_dim);
    auto broadcasted_gamma =
        rewriter.create<ConstOp>(loc, broadcasted_gamma_attr);

    // Inject a mul between the filter constant and the quantize op.
    auto new_filter = rewriter
                          .create<TFL::MulOp>(loc, filter, broadcasted_gamma,
                                              rewriter.getStringAttr("NONE"))
                          .getResult();
    // Update the scale in the quantize op.
    auto new_qtype = RescaleQtype(q_op.getQtype(), gamma_cst);
    if (!new_qtype) {
      return failure();
    }
    // Update the Q op to a new Q op with the new multiplied scale.
    rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(q_op, new_qtype.getValue(),
                                                 new_filter, new_qtype);
    // If bias isn't None, it needs to be multiplied as well.
    if (!mlir::isa<NoneType>(bias.getType())) {
      rewriter.setInsertionPoint(affine_op);

      auto squeezed_gamma = FlattenTo1D(gamma_cst);
      auto squeezed_gamma_type = squeezed_gamma.getType();
      auto squeezed_gamma_op = rewriter.create<arith::ConstantOp>(
          affine_op.getLoc(), squeezed_gamma_type, squeezed_gamma);

      auto new_bias = rewriter.create<TFL::MulOp>(
          loc, bias, squeezed_gamma_op, rewriter.getStringAttr("NONE"));
      affine_op.getOperation()->replaceUsesOfWith(bias, new_bias);
    }

    // Remove the tailing mul op.
    mul_op.replaceAllUsesWith(affine_op.getResult());
    return success();
  }
};

using FuseConv2DAndMulWithQDQs = FuseAffinOpAndMulWithQDQs<TFL::Conv2DOp>;
using FuseDepthwiseConv2DAndMulWithQDQs =
    FuseAffinOpAndMulWithQDQs<TFL::DepthwiseConv2DOp>;

// Fuse Binary Op with following Affine operation.
template <typename AffineOpType>
struct FuseBinaryOpToFollowingAffineOp : public OpRewritePattern<AffineOpType> {
  using OpRewritePattern<AffineOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineOpType fc_op,
                                PatternRewriter &rewriter) const override {
    // Binary op.
    Operation *binary_op = fc_op.getInput().getDefiningOp();
    if (!binary_op || binary_op->getNumOperands() != 2) return failure();
    // We only handle the cases the RHS is a scalar.
    // TODO(fengliuai): Currently the canonicalizer pass couldn't guarantee that
    // the constant operands are on the RHS, we need to consider LHS constant
    // operand if necessary.
    DenseFPElementsAttr cst;
    if (!matchPattern(binary_op->getOperand(1), m_Constant(&cst)))
      return failure();
    if (cst.getNumElements() != 1) return failure();
    APFloat cst_value = *cst.value_begin<APFloat>();

    // Affine op.
    Value filter = fc_op.getFilter();
    Value bias = fc_op.getBias();
    DenseFPElementsAttr filter_cst, bias_cst;
    if (!matchPattern(filter, m_Constant(&filter_cst))) {
      // The filter maybe quantized, then we should set it to the real constant.
      auto dq = llvm::dyn_cast_or_null<DequantizeOp>(filter.getDefiningOp());
      if (!dq) return failure();
      auto q =
          llvm::dyn_cast_or_null<QuantizeOp>(dq.getInput().getDefiningOp());
      if (!q || !matchPattern(q.getInput(), m_Constant(&filter_cst))) {
        return failure();
      }
      filter = q.getInput();
    }
    if (!mlir::isa<NoneType>(bias.getType()) &&
        !matchPattern(bias, m_Constant(&bias_cst)))
      return failure();
    auto binary_op_activation_func =
        binary_op->template getAttrOfType<StringAttr>(
            "fused_activation_function");
    if (!binary_op_activation_func ||
        binary_op_activation_func.getValue() != "NONE")
      return failure();
    ShapedType filter_type = filter_cst.getType();

    if (llvm::isa<AddOp, SubOp>(binary_op)) {
      auto padding = fc_op->template getAttrOfType<StringAttr>("padding");
      if (padding && padding.getValue() != "VALID") return failure();

      // The fusion of add/sub is actually applying the following
      // transformation:
      // w * (x + c) + b => w * x + (w * c + b)
      // so we have to update the bias.
      if (llvm::isa<SubOp>(binary_op)) cst_value.changeSign();

      auto bias_and_slice =
          GetBiasDimAndSliceSize(filter_type.getShape(), fc_op);
      int64_t bias_size = bias_and_slice.first;
      int64_t slice_size = bias_and_slice.second;
      ShapedType new_bias_type =
          RankedTensorType::get({bias_size}, filter_type.getElementType());

      // The new bias should be a 1-D tensor with length equals to the bias
      // dimension of the weight.
      SmallVector<APFloat, 4> new_bias_values;
      if (mlir::isa<NoneType>(bias.getType())) {  // none bias, a list of zeros
        new_bias_values.resize(bias_size,
                               APFloat::getZero(cst_value.getSemantics()));
      } else if (bias_cst.getNumElements() == 1) {  // scalar bias, broadcast it
        new_bias_values.resize(bias_size, *bias_cst.value_begin<APFloat>());
      } else if (bias_cst.getNumElements() == bias_size) {  // 1-d bias, copy it
        new_bias_values.insert(new_bias_values.begin(),
                               bias_cst.value_begin<APFloat>(),
                               bias_cst.value_end<APFloat>());
      } else {
        return failure();
      }

      int64_t flatten_index = 0;
      for (auto fp_it = filter_cst.value_begin<APFloat>(),
                fp_end = filter_cst.value_end<APFloat>();
           fp_it != fp_end; ++fp_it) {
        int bias_index = (flatten_index++ / slice_size) % bias_size;

        new_bias_values[bias_index] =
            new_bias_values[bias_index] + *fp_it * cst_value;
      }
      auto new_bias = DenseFPElementsAttr::get(new_bias_type, new_bias_values);
      auto new_bias_op =
          rewriter.create<ConstOp>(fc_op.getLoc(), new_bias_type, new_bias);
      fc_op.setOperand(0, binary_op->getOperand(0));
      fc_op.setOperand(2, new_bias_op);
    } else if (llvm::isa<MulOp, DivOp>(binary_op)) {
      // The fusion of mul/div is actually applying the following
      // transformation:
      // w * (x ' c) + b => (w ' c) x + b
      // so we have to update the weight.
      bool is_mul = llvm::isa<MulOp>(binary_op);
      auto new_filter =
          filter_cst.mapValues(filter_type.getElementType(), [&](APFloat it) {
            return (is_mul ? it * cst_value : it / cst_value).bitcastToAPInt();
          });
      // We recreate the constant op in case it is shared by the other ops. This
      // might increase the model size.
      auto new_filter_op = rewriter.create<ConstOp>(
          fc_op.getLoc(), filter.getType(), new_filter);
      fc_op.setOperand(0, binary_op->getOperand(0));
      if (fc_op.getFilter() != filter) {
        // This filter goes through quantize and dequantize ops. Then we just
        // need to update the weight to the quantize op.
        filter.replaceAllUsesWith(new_filter_op);
      } else {
        // This filter doesn't go through quantize and dequantize ops, Then
        // we update the weight of the affine op directly.
        fc_op.setOperand(1, new_filter_op);
      }
    } else {
      return failure();
    }
    return success();
  }

 private:
  // Returns the dimension length of the channel dimension and also the slide
  // size by each position in the channel dimension accordingly. tfl.conv2d and
  // tfl.fully_connected has heading channel dimension, but tfl.depthwise_conv2d
  // has tailing channel dimension. This function is to provide a utility to
  // create the above information from the op property.
  static std::pair<int64_t, int64_t> GetBiasDimAndSliceSize(
      ArrayRef<int64_t> filter_shape, AffineOpType op) {
    // Channel dimension index is specified as op property
    auto channel_index_iter = filter_shape.begin();
    std::advance(channel_index_iter, op.GetChannelDimIndex());
    // The slide size is the size of the data in higher dimensions.
    int64_t slice_size =
        std::accumulate(std::next(channel_index_iter), filter_shape.end(), 1,
                        std::multiplies<int64_t>());
    return {*channel_index_iter, slice_size};
  }
};

// Remove Reshape before FullyConnected when `keep_num_dims=false` and Reshape
// does not alter the last dimension as FullyConnected will collapse all other
// dimensions into a single dimension. For example,
//
//   %shape = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
//   %reshape = tfl.reshape(%input, %shape) // %input: tensor<128x64xf32>
//   %fc = tfl.fully_connected(%reshape, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
struct RemoveReshapeBeforeFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fully_connected_op,
                                PatternRewriter &) const override {
    auto input = fully_connected_op.getInput();
    auto input_ty = mlir::dyn_cast<ShapedType>(input.getType());
    auto output_ty =
        mlir::dyn_cast<ShapedType>(fully_connected_op.getOutput()[0].getType());
    if (!input_ty.hasStaticShape() ||
        fully_connected_op.getWeightsFormat() != "DEFAULT" ||
        fully_connected_op.getKeepNumDims() || !output_ty.hasStaticShape() ||
        output_ty.getRank() != 2) {
      return failure();
    }

    auto reshape_op = input.getDefiningOp<TFL::ReshapeOp>();
    if (!reshape_op) return failure();

    // Check if the last dimension does not change after reshape.
    auto reshape_input = reshape_op.getInput();
    auto reshape_input_ty = mlir::dyn_cast<ShapedType>(reshape_input.getType());
    if (!reshape_input_ty.hasStaticShape() || input_ty.getRank() == 0 ||
        reshape_input_ty.getRank() == 0 ||
        input_ty.getDimSize(input_ty.getRank() - 1) !=
            reshape_input_ty.getDimSize(reshape_input_ty.getRank() - 1)) {
      return failure();
    }

    // Connect the input to the one of reshape.
    fully_connected_op.setOperand(0, reshape_input);
    return success();
  }
};

// Remove Reshape after FullyConnected when `keep_num_dims=false`, the Reshape
// does not alter the last dimension and it restores the batch dimensions
// collapsed by the FullyConnected op due to `keep_num_dims=false`. For example,
//
//   // %input: tensor<4x16x32xf32>
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//   %shape = arith.constant dense<[4, 16, 32]> : tensor<3xi32>
//   %rs = tfl.reshape(%fc, %shape)
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = true, weights_format = "DEFAULT"}
struct RemoveReshapeAfterFullyConnected
    : public OpRewritePattern<TFL::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape_op,
                                PatternRewriter &rewriter) const override {
    auto fully_connected_op = llvm::dyn_cast_or_null<TFL::FullyConnectedOp>(
        reshape_op.getInput().getDefiningOp());
    if (!fully_connected_op || fully_connected_op.getNumResults() != 1 ||
        fully_connected_op.getWeightsFormat() != "DEFAULT" ||
        fully_connected_op.getKeepNumDims())
      return failure();
    if (!reshape_op.getInput().hasOneUse()) return failure();

    auto input_shape =
        mlir::cast<ShapedType>(fully_connected_op.getInput().getType());
    auto output_shape = mlir::cast<ShapedType>(fully_connected_op.getType(0));
    auto reshape_shape = mlir::cast<ShapedType>(reshape_op.getType());
    if (!input_shape.hasStaticShape() || !output_shape.hasStaticShape() ||
        !reshape_shape.hasStaticShape())
      return failure();

    // Check that the reshape doesn't modify the last dimension and it restores
    // the input (batch) dimension with the exception of the feature (last)
    // dimension.
    if (output_shape.getShape().empty() || reshape_shape.getShape().empty() ||
        output_shape.getShape().back() != reshape_shape.getShape().back() ||
        input_shape.getShape().drop_back() !=
            reshape_shape.getShape().drop_back())
      return failure();

    llvm::SmallVector<Type, 1> output_type{reshape_op.getType()};
    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        reshape_op, output_type, /*input=*/fully_connected_op.getInput(),
        /*filter=*/fully_connected_op.getFilter(),
        /*bias=*/fully_connected_op.getBias(),
        /*fused_activation_function=*/
        fully_connected_op.getFusedActivationFunction(),
        /*weights_format=*/fully_connected_op.getWeightsFormat(),
        /*keep_num_dims=*/true,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.getAsymmetricQuantizeInputsAttr());
    return success();
  }
};

// Fuses Unpack with proceeding Concatenation to Reshape if output type has
// static shape and activation function is none. For example:
//
//   // %input: tensor<1x3x2xf32>
//   %unpack:3 = "tfl.unpack"(%input) {axis = 1 : i32, num = 3 : i32}
//   %res = "tfl.concatenation"(%unpack#0, %unpack#1, %unpack#2)
//        {axis = -1 : i32, fused_activation_function = "NONE"}
//
// can be optimized to
//
//   %cst = arith.constant dense<[1, 6]> : tensor<2xi32>
//   %res = "tfl.reshape"(%input, %cst)
struct FuseUnpackAndConcatToReshape
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concat_op,
                                PatternRewriter &rewriter) const override {
    if (concat_op.getFusedActivationFunction() != "NONE") {
      return failure();
    }

    // Checks all operands come from the same unpack op.
    auto first_operand = concat_op.getValues().front();
    auto unpack_op =
        dyn_cast_or_null<TFL::UnpackOp>(first_operand.getDefiningOp());
    if (!unpack_op || unpack_op.getNumResults() != concat_op.getNumOperands()) {
      return failure();
    }
    for (const auto &index_and_value : llvm::enumerate(concat_op.getValues())) {
      if (index_and_value.value() !=
          unpack_op.getResult(index_and_value.index())) {
        return failure();
      }
    }

    auto output_type = mlir::cast<ShapedType>(concat_op.getType());
    if (!output_type.hasStaticShape()) {
      return failure();
    }

    auto new_shape_array = output_type.getShape();
    // This is to workaround the unnecessary cast i64 -> i32.
    SmallVector<int32_t, 4> new_shape_array_i32;
    for (auto size : new_shape_array) {
      new_shape_array_i32.push_back(
          ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size));
    }
    auto new_shape = rewriter.create<TFL::ConstOp>(
        concat_op.getLoc(), GetI32ElementsAttr(new_shape_array_i32, &rewriter));

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        concat_op, output_type, unpack_op.getInput(), new_shape);
    return success();
  }
};

// Reduce the K of a TopKV2Op for the following case.
//
// values, indices = tfl.topkv2(%inputs, K)
// %1 = tfl.slice(values, 0, k)
// %2 = tfl.slice(indices,0, k)
// .... (values and indices only used for %1 and %2)
//
// %1 or %2 can be absent. If values and indices are only used here,
// this pattern can be replaced with (conceptually)
//
// %values, %indices = tfl.topkv2(%inputs, k)
// replace all use of %1 with values
// replace all use of %2 with indices
//
struct OptimizeTopK : public OpRewritePattern<TFL::TopKV2Op> {
  using OpRewritePattern::OpRewritePattern;

  // It computes the last dim k of slice size of value.user.
  // If value has no use then return 0.
  std::optional<int32_t> ComputeSliceK(Value value) const {
    if (value.use_empty()) return 0;
    auto slice_op =
        llvm::dyn_cast_or_null<TFL::SliceOp>(value.getUses().begin().getUser());
    // We only match for the case where value is used by SliceOp.
    if (!slice_op) return std::nullopt;
    DenseElementsAttr begin;
    DenseElementsAttr size;
    if (!matchPattern(slice_op->getOperand(1), m_Constant(&begin)) ||
        !matchPattern(slice_op->getOperand(2), m_Constant(&size)))
      return std::nullopt;

    // Check if "begin" is a zero tensor.
    for (auto begin_idx : begin.getValues<APInt>())
      if (begin_idx != 0) return std::nullopt;

    // Check if "size" is equal to slice_op.input.shape except
    // for last dimension.
    // It can be done  by verifying the number of elements:
    // i.e., num_input/input_last_dim = num_result/k
    auto input_ty = mlir::dyn_cast_or_null<ShapedType>(value.getType());
    auto result_ty = mlir::dyn_cast<ShapedType>(slice_op.getType());
    if (!input_ty || !result_ty) return std::nullopt;
    if (!input_ty.hasStaticShape() || !result_ty.hasStaticShape())
      return std::nullopt;
    if (!input_ty.getRank() || !result_ty.getRank()) return std::nullopt;
    int num_input = input_ty.getNumElements();
    int input_last_dim = input_ty.getShape().back();
    if (input_last_dim < 1) return std::nullopt;
    int num_result = result_ty.getNumElements();
    auto size_last = *(--size.value_end<APInt>());
    int32_t k = size_last.getSExtValue();
    if (num_input / input_last_dim * k != num_result) return std::nullopt;
    // We don't match sliceOp with last dim size = 0.
    if (!k) return std::nullopt;
    return k;
  }

  LogicalResult matchAndRewrite(TFL::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
    auto values = op.getValues();
    auto indices = op.getIndices();
    // op.getValues() and op.getIndices() cannot be used more than once.
    if (!values.hasOneUse() && !values.use_empty()) return failure();
    if (!indices.hasOneUse() && !indices.use_empty()) return failure();

    auto k_values_or = ComputeSliceK(values);
    auto k_indices_or = ComputeSliceK(indices);
    if (!k_values_or.has_value() || !k_indices_or.has_value()) return failure();
    int32_t k_values = k_values_or.value();
    int32_t k_indices = k_indices_or.value();
    // We don't match two SliceOp with different sizes.
    if (k_values != k_indices && !values.use_empty() && !indices.use_empty())
      return failure();

    // Start replacing.
    auto k = !values.use_empty() ? k_values : k_indices;
    // Build scalar tensor k.
    auto k_ty = mlir::RankedTensorType::get({}, rewriter.getIntegerType(32));
    Value k_cst = rewriter.create<TFL::ConstOp>(
        op.getLoc(), DenseElementsAttr::get(k_ty, k));
    // Compute new result types.
    auto values_ty = mlir::dyn_cast<ShapedType>(values.getType());
    auto indices_ty = mlir::dyn_cast<ShapedType>(indices.getType());
    auto shape = std::vector<int64_t>();
    for (auto d : values_ty.getShape().drop_back()) {
      shape.push_back(d);
    }
    shape.push_back(static_cast<int64_t>(k));
    auto new_values_ty =
        mlir::RankedTensorType::get(shape, values_ty.getElementType());
    auto new_indices_ty =
        mlir::RankedTensorType::get(shape, indices_ty.getElementType());
    TFL::TopKV2Op top_k_op = rewriter.create<TFL::TopKV2Op>(
        op.getLoc(), new_values_ty, new_indices_ty, op->getOperand(0), k_cst);

    // Remove original ops (topk, Slice, Slice).
    if (!values.use_empty()) {
      auto values_slice_op = llvm::dyn_cast_or_null<TFL::SliceOp>(
          values.getUses().begin().getUser());
      values_slice_op.getResult().replaceAllUsesWith(top_k_op.getValues());
      values_slice_op.erase();
    }
    if (!indices.use_empty()) {
      auto indices_slice_op = llvm::dyn_cast_or_null<TFL::SliceOp>(
          indices.getUses().begin().getUser());
      indices_slice_op.getResult().replaceAllUsesWith(top_k_op.getIndices());
      indices_slice_op.erase();
    }
    op.erase();
    return success();
  }
};

using FuseBinaryOpToFollowingFullyConnected =
    FuseBinaryOpToFollowingAffineOp<FullyConnectedOp>;
using FuseBinaryOpToFollowingDepthwiseConv2D =
    FuseBinaryOpToFollowingAffineOp<DepthwiseConv2DOp>;
using FuseBinaryOpToFollowingConv2D = FuseBinaryOpToFollowingAffineOp<Conv2DOp>;

// Optimizes transpose->reshape->batch_matmul->reshape->transpose to a single
// batch_matmul.
struct FuseReshapeAndTransposeAroundBatchMatmul
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    TensorType transpose_input_type = op.getInput().getType();
    // TODO(chhe): to support more than 3D in this pattern.
    if (!transpose_input_type.hasStaticShape() ||
        transpose_input_type.getRank() != 3) {
      return failure();
    }
    DenseIntElementsAttr transpose_perm;
    if (!matchPattern(op.getPerm(), m_Constant(&transpose_perm))) {
      return failure();
    }
    const SmallVector<int64_t, 3> match_perm = {1, 2, 0};
    for (const auto &[perm_index, match_perm_index] :
         llvm::zip(transpose_perm.getValues<APInt>(), match_perm)) {
      if (perm_index != match_perm_index) {
        return failure();
      }
    }

    auto reshape_op = op.getInput().getDefiningOp<ReshapeOp>();
    if (!reshape_op ||
        !InsertOneInSecondInnermostDim(reshape_op.getInput().getType(),
                                       reshape_op.getType())) {
      return failure();
    }

    auto batch_matmul = reshape_op.getInput().getDefiningOp<BatchMatMulOp>();
    if (!batch_matmul || batch_matmul.getAdjY()) {
      return failure();
    }

    auto reshape_op_1 = batch_matmul.getY().getDefiningOp<ReshapeOp>();
    if (!reshape_op_1 ||
        !InsertOneInSecondInnermostDim(reshape_op_1.getType(),
                                       reshape_op_1.getInput().getType())) {
      return failure();
    }

    auto transpose_op = reshape_op_1.getInput().getDefiningOp<TransposeOp>();
    if (!transpose_op) {
      return failure();
    }
    DenseIntElementsAttr transpose_perm_1;
    if (!matchPattern(transpose_op.getPerm(), m_Constant(&transpose_perm_1)) ||
        !TransposeFirstTwoDimToLast(transpose_perm_1)) {
      return failure();
    }

    TypedValue<TensorType> transpose_input = transpose_op.getInput();
    SmallVector<int, 3> new_shape = {
        static_cast<int>(transpose_input.getType().getDimSize(0)),
        static_cast<int>(transpose_input.getType().getDimSize(1)),
        static_cast<int>(std::accumulate(
            transpose_input.getType().getShape().begin() + 2,
            transpose_input.getType().getShape().end(), 1, std::multiplies()))};
    auto shape_constant = rewriter.create<ConstOp>(
        batch_matmul.getLoc(), GetI32ElementsAttr(new_shape, &rewriter));
    auto reshaped_input = rewriter.create<ReshapeOp>(
        batch_matmul.getLoc(), transpose_op.getInput(), shape_constant);
    rewriter.replaceOpWithNewOp<BatchMatMulOp>(
        op, op.getType(), reshaped_input, batch_matmul.getX(),
        /*adj_x=*/false, /*adj_y=*/!batch_matmul.getAdjX(),
        batch_matmul.getAsymmetricQuantizeInputsAttr());
    return success();
  }

 private:
  // Checks that tensor `a` has shape of [M, N] and `b` has
  // [M_0, M_1, ..., 1, N], where `M = M_0 * M_1 * ...`.
  bool InsertOneInSecondInnermostDim(TensorType a, TensorType b) const {
    if (!a.hasStaticShape() || !b.hasStaticShape()) return false;
    if (a.getRank() != 2 || b.getRank() < 2) return false;
    if (a.getShape().back() != b.getShape().back()) return false;
    return b.getDimSize(b.getRank() - 2) == 1;
  }

  // Checks if the transpose permutation has value [2, 3, ..., n-1, 0, 1].
  bool TransposeFirstTwoDimToLast(DenseIntElementsAttr perm) const {
    int rank = perm.getNumElements();
    if (rank < 3) return false;
    for (int i = 0; i < rank - 2; i++) {
      if (perm.getValues<APInt>()[i] != i + 2) {
        return false;
      }
    }
    return perm.getValues<APInt>()[rank - 2] == 0 &&
           perm.getValues<APInt>()[rank - 1] == 1;
  }
};

// Optimizes transpose->reshape->batch_matmul to reshape->batch_matmul.
struct FuseTransposeReshapeIntoBatchMatmul
    : public OpRewritePattern<TFL::BatchMatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::BatchMatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto reshape_op = op.getY().getDefiningOp<ReshapeOp>();
    if (!reshape_op || !ReshapeFirstTwoDim(reshape_op.getInput().getType(),
                                           reshape_op.getType())) {
      return failure();
    }

    auto transpose_op = reshape_op.getInput().getDefiningOp<TransposeOp>();
    if (!transpose_op) {
      return failure();
    }
    DenseIntElementsAttr transpose_perm;
    if (!matchPattern(transpose_op.getPerm(), m_Constant(&transpose_perm)) ||
        !TransposeLastTwoDimToFirst(transpose_perm)) {
      return failure();
    }

    SmallVector<int> new_shape(
        reshape_op.getType().getShape().drop_front().begin(),
        reshape_op.getType().getShape().drop_front().end());
    new_shape.push_back(reshape_op.getType().getDimSize(0));
    auto shape_constant = rewriter.create<ConstOp>(
        op.getLoc(), GetI32ElementsAttr(new_shape, &rewriter));
    auto new_reshape = rewriter.create<ReshapeOp>(
        op.getLoc(), transpose_op.getInput(), shape_constant);
    rewriter.replaceOpWithNewOp<BatchMatMulOp>(
        op, op.getType(), op.getX(), new_reshape, op.getAdjX(), !op.getAdjY(),
        op.getAsymmetricQuantizeInputsAttr());
    return success();
  }

 private:
  // Checks that tensor `a` has shape of [M, N, ...] and `b` has [M * N, ...].
  bool ReshapeFirstTwoDim(TensorType a, TensorType b) const {
    if (!a.hasStaticShape() || !b.hasStaticShape()) return false;
    if (a.getRank() < 2 || b.getRank() < 1) return false;
    return a.getShape().drop_front(2) == b.getShape().drop_front(1);
  }

  // Checks if the transpose permutation has value [n-2, n-1, 0, 1, ...].
  bool TransposeLastTwoDimToFirst(DenseIntElementsAttr perm) const {
    const int rank = perm.getNumElements();
    auto perm_iter = perm.getValues<APInt>();
    if (rank < 3) return false;
    for (int i = 2; i < rank; i++) {
      if (perm_iter[i] != i - 2) {
        return false;
      }
    }
    return perm_iter[0] == rank - 2 && perm_iter[1] == rank - 1;
  }
};

struct FuseLogSoftmax : public OpRewritePattern<TFL::SubOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::SubOp sub_op,
                                PatternRewriter &rewriter) const override {
    if (sub_op.getFusedActivationFunction() != "NONE") {
      return failure();
    }
    auto log_op = dyn_cast_or_null<TFL::LogOp>(sub_op.getRhs().getDefiningOp());
    if (!log_op || !log_op->hasOneUse()) {
      return failure();
    }
    auto sum_op = dyn_cast_or_null<TFL::SumOp>(log_op.getX().getDefiningOp());
    if (!sum_op || !sum_op.getKeepDims() ||
        !isSupportedAxis(
            sum_op.getAxes(),
            mlir::cast<ShapedType>(sum_op.getOperand(0).getType()).getRank())) {
      return failure();
    }
    if (!sum_op->hasOneUse()) {
      return failure();
    }
    auto exp_op =
        dyn_cast_or_null<TFL::ExpOp>(sum_op.getInput().getDefiningOp());
    if (!exp_op || !exp_op->hasOneUse()) {
      return failure();
    }

    auto parent_sub_op =
        dyn_cast_or_null<TFL::SubOp>(sub_op.getLhs().getDefiningOp());
    if (!parent_sub_op || parent_sub_op != dyn_cast_or_null<TFL::SubOp>(
                                               exp_op.getX().getDefiningOp())) {
      return failure();
    }
    if (std::distance(parent_sub_op->getUses().begin(),
                      parent_sub_op->getUses().end()) != 2) {
      return failure();
    }

    auto reduce_max_op = dyn_cast_or_null<TFL::ReduceMaxOp>(
        parent_sub_op.getRhs().getDefiningOp());
    if (!reduce_max_op || !reduce_max_op->hasOneUse() ||
        !reduce_max_op.getKeepDims() ||
        !isSupportedAxis(
            reduce_max_op.getAxes(),
            mlir::cast<ShapedType>(reduce_max_op.getOperand(0).getType())
                .getRank())) {
      return failure();
    }

    if (reduce_max_op.getInput() != parent_sub_op.getLhs()) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<TFL::LogSoftmaxOp>(sub_op, sub_op.getType(),
                                                   parent_sub_op.getLhs());
    return success();
  }

  // The TFL_LogSoftmaxOp implementation only works on the last axis, so we
  // check that both TFL_ReduceMaxOP and TFL_SumOp use the last axis
  bool isSupportedAxis(mlir::Value value, int64_t rank) const {
    auto const_op =
        dyn_cast_or_null<mlir::arith::ConstantOp>(value.getDefiningOp());
    if (!const_op) {
      return false;
    }
    auto axes = dyn_cast<DenseIntElementsAttr>(const_op.getValueAttr());
    if (!axes || axes.getNumElements() != 1) {
      return false;
    }
    auto axes_elem_ty = axes.getType().getElementType();
    if (!axes_elem_ty.isInteger(32) && !axes_elem_ty.isInteger(64)) {
      return false;
    }
    const int64_t axis = (*axes.begin()).getSExtValue();
    if (axis != rank - 1 && axis != -1) {
      return false;
    }
    return true;
  }
};

// This is maintained for backward compatibility but not included in the strict
// QDQ mode.
// Equivalent to:
// def eliminate_dq_q_pairs : Pat<
//   (TFL_QuantizeOp (TFL_DequantizeOp $in), $qt),
//   (replaceWithValue $in),
//   [(NotFromQuantOpOrSameQuantType $in, $qt)]>;
struct EliminateQDQPairs : public OpRewritePattern<TFL::QuantizeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TFL::QuantizeOp q_op,
                                PatternRewriter &rewriter) const override {
    if (auto dq_op = dyn_cast_or_null<TFL::DequantizeOp>(
            q_op.getInput().getDefiningOp())) {
      if (tflite::NotFromQuantOpOrSameQuantType(dq_op.getInput(),
                                                q_op.getQtypeAttr())) {
        q_op.replaceAllUsesWith(dq_op.getInput());
        return success();
      }
      return rewriter.notifyMatchFailure(q_op,
                                         "preceding DequantizeOp has different "
                                         "input than the quantize quant type.");
    }
    return rewriter.notifyMatchFailure(q_op, "not preceded by a DequantizeOp.");
  }
};

// This is the UndoBroadcastFullyConnectedBiasAdd pattern in
// optimize_patterns.td but accounting for QDQ preceding Add's RHS.
// The following doesn't work in TableGen due to some issues reconstructing
// TFL_DequantizeOp.
// def UndoBroadcastFullyConnectedBiasAddWithQDQs : Pat<
//   (TFL_AddOp $lhs,
//     (TFL_DequantizeOp
//       (TFL_QuantizeOp
//         (Arith_ConstantOp:$const_op $bias),
//       $qparams)),
//   $act_fn),
//   (TFL_AddOp $lhs,
//     (TFL_DequantizeOp
//       (TFL_QuantizeOp
//         (Arith_ConstantOp:$const_op (FlattenTo1D $bias),
//       $qparams)),
//   $act_fn),
//   [(AnyStaticShapeTensor $lhs),
//    (IsLastDimEqualToNumElements $bias, $bias),
//    (HasOneUse $const_op),
//    (HasRankAtMost<4> $bias),
//    (HasRankAtLeast<2> $bias),
//    (IsDefinedByFullyConnectedOp $lhs)]>;
struct UndoBroadcastFullyConnectedBiasAddWithQDQs
    : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp add_op,
                                PatternRewriter &rewriter) const override {
    if (!add_op->hasOneUse()) {
      return failure();
    }

    auto fc_op = dyn_cast_or_null<TFL::FullyConnectedOp>(
        add_op.getLhs().getDefiningOp());
    if (!fc_op) {
      return failure();
    }

    auto dq_op =
        dyn_cast_or_null<TFL::DequantizeOp>(add_op.getRhs().getDefiningOp());
    if (!dq_op) {
      return failure();
    }

    auto q_op =
        dyn_cast_or_null<TFL::QuantizeOp>(dq_op.getInput().getDefiningOp());
    if (!q_op) {
      return failure();
    }

    auto bias_op =
        dyn_cast_or_null<arith::ConstantOp>(q_op.getInput().getDefiningOp());
    if (!bias_op) {
      return failure();
    }

    auto bias_type = bias_op.getType();
    auto bias_rank = mlir::cast<ShapedType>(bias_type).getRank();
    if (bias_rank > 4 || bias_rank < 2) {
      return failure();
    }

    if (!IsLastDimEqualToNumElements(bias_type, bias_type)) {
      return failure();
    }

    auto new_bias = FlattenTo1D(bias_op.getValueAttr());
    auto new_bias_type = new_bias.getType();
    auto new_bias_op = rewriter.create<arith::ConstantOp>(
        bias_op.getLoc(), new_bias_type, new_bias);

    // Update QuantizeOp with the new bias and its output shape
    q_op.setOperand(new_bias_op);
    auto new_q_op_type =
        RankedTensorType::Builder(
            mlir::cast<RankedTensorType>(q_op.getResult().getType()))
            .setShape(mlir::cast<ShapedType>(new_bias_type).getShape());
    q_op.getResult().setType(new_q_op_type);
    auto attr = TypeAttr::get(q_op.getResult().getType());
    q_op.setQtypeAttr(attr);

    // Update DequantizeOp's output shape
    auto new_dq_op_type =
        RankedTensorType::Builder(
            mlir::cast<RankedTensorType>(dq_op.getResult().getType()))
            .setShape(mlir::cast<ShapedType>(new_bias_type).getShape());
    dq_op.getResult().setType(new_dq_op_type);

    // Remove old bias
    rewriter.eraseOp(bias_op);
    return success();
  }
};

// Move Reshape after FullyConnected to before FullyConnected when possible.
// For some cases where Reshape-FC-Reshape pattern can not be fused, moving
// Reshape after to before FC may help remove one Reshape op by folding
// (Reshape-Reshape)-FC.
struct MoveReshapeAfterFullyConnected
    : public OpRewritePattern<TFL::ReshapeOp> {
  explicit MoveReshapeAfterFullyConnected(MLIRContext *context)
      : OpRewritePattern<TFL::ReshapeOp>(context, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    auto fc = llvm::dyn_cast_or_null<TFL::FullyConnectedOp>(
        reshape.getInput().getDefiningOp());

    if (!fc || fc.getNumResults() != 1 || !fc.getResult(0).hasOneUse()) {
      return failure();
    }
    if (auto before = fc.getInput().getDefiningOp();
        !before || !mlir::isa<TFL::ReshapeOp>(before)) {
      return failure();
    }

    auto input_ty =
        mlir::dyn_cast_or_null<RankedTensorType>(fc.getInput().getType());
    auto fc_ty = mlir::dyn_cast_or_null<RankedTensorType>(fc.getType(0));
    auto reshape_ty =
        mlir::dyn_cast_or_null<RankedTensorType>(reshape.getResult().getType());
    if (!input_ty || !fc_ty || !reshape_ty || !input_ty.hasStaticShape() ||
        !fc_ty.hasStaticShape() || !reshape_ty.hasStaticShape()) {
      return failure();
    }

    if (reshape_ty.getRank() < 2 ||
        reshape_ty.getShape().back() != fc_ty.getShape().back()) {
      // The movable Reshape after must satisfy:
      // 1. Reshape output's rank >= 2. (FC does not support 1D tensor input).
      // 2. FC and Reshape outputs' shape are both (..., N).
      return failure();
    }

    llvm::SmallVector<int32_t> new_input_shape(reshape_ty.getShape());
    new_input_shape.pop_back();
    new_input_shape.push_back(input_ty.getShape().back());

    auto reshape_before = rewriter.create<TFL::ReshapeOp>(
        fc.getLoc(), fc.getInput(),
        rewriter.create<arith::ConstantOp>(
            fc->getLoc(), GetI32ElementsAttr(new_input_shape, &rewriter)));

    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        reshape,
        RankedTensorType::get(reshape_ty.getShape(),
                              reshape_ty.getElementType()),
        reshape_before, fc.getFilter(), fc.getBias(),
        fc.getFusedActivationFunction(), fc.getWeightsFormat(),
        /*keep_num_dims=*/true, fc.getAsymmetricQuantizeInputsAttr());
    return success();
  }
};

// When FullyConnected is followed by a Reshape op, the shape of the
// FullyConnected's output doesn't matter. Enabling FC's keep_num_dims in such
// case is valid and may help downstream runtime e.g. GPU delegate do better
// layout planning.
struct EnableFullyConnectedKeepNumDimsBeforeReshape
    : public OpRewritePattern<TFL::ReshapeOp> {
  explicit EnableFullyConnectedKeepNumDimsBeforeReshape(MLIRContext *context)
      : OpRewritePattern<TFL::ReshapeOp>(context, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    auto fc = llvm::dyn_cast_or_null<TFL::FullyConnectedOp>(
        reshape.getInput().getDefiningOp());

    if (!fc || fc.getNumResults() != 1 || fc.getKeepNumDims() ||
        !fc->hasOneUse()) {
      return failure();
    }

    auto input_ty =
        mlir::dyn_cast_or_null<RankedTensorType>(fc.getInput().getType());
    auto fc_ty = mlir::dyn_cast_or_null<RankedTensorType>(fc.getType(0));
    if (!input_ty || !fc_ty || input_ty.getRank() == 2) {
      return failure();
    }

    llvm::SmallVector<int64_t> new_fc_shape(input_ty.getShape());
    new_fc_shape.pop_back();
    new_fc_shape.push_back(fc_ty.getShape().back());

    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        fc, RankedTensorType::get(new_fc_shape, fc_ty.getElementType()),
        fc.getInput(), fc.getFilter(), fc.getBias(),
        fc.getFusedActivationFunction(), fc.getWeightsFormat(),
        /*keep_num_dims=*/true, fc.getAsymmetricQuantizeInputsAttr());
    return success();
  }
};

// This pattern push transposes through squeeze ops to facilitate further
// transpose and reshape fusions. For example, some JAX model could have
// subgraphs like Reshape-Transpose-Squeeze. With this pattern, the transpose
// can be pushed through the squeeze op, and fused with a subsequent reshape or
// removed entirely. The squeeze op could also be fused with the former reshape.
//
// The pattern is designed to have lower benefit/priority than others,
// while the push may still happen if the transpose could be fused with
// downstream optimization phases or passe..
struct PushTransposeThroughSqueeze : public RewritePattern {
  explicit PushTransposeThroughSqueeze(MLIRContext *context)
      : RewritePattern(TFL::SqueezeOp::getOperationName(), /*benefit=*/0,
                       context) {}

  LogicalResult matchAndRewrite(mlir::Operation *op,
                                PatternRewriter &rewriter) const override {
    TFL::SqueezeOp squeeze = cast<TFL::SqueezeOp>(op);
    auto transpose = llvm::dyn_cast_or_null<TFL::TransposeOp>(
        squeeze.getInput().getDefiningOp());
    if (!transpose) {
      return failure();
    }

    int32_t input_rank = transpose.getType().getShape().size();

    llvm::SmallVector<int32_t, 4> squeeze_dims;
    if (squeeze->hasAttr("squeeze_dims")) {
      for (const auto &squeeze_dim : squeeze.getSqueezeDimsAttr()) {
        squeeze_dims.push_back(
            mlir::dyn_cast<IntegerAttr>(squeeze_dim).getInt());
      }
    }
    if (squeeze_dims.empty()) {
      for (int dim = 0; dim < input_rank; ++dim) {
        if (transpose.getType().getDimSize(dim) == 1) {
          squeeze_dims.push_back(dim);
        }
      }
    }

    mlir::DenseIntElementsAttr perm_attr;
    if (!matchPattern(transpose.getPerm(), m_Constant(&perm_attr))) {
      return failure();
    }
    llvm::SmallVector<int32_t, 4> perm;
    for (const auto &dim : perm_attr.getValues<APInt>()) {
      perm.push_back(dim.getSExtValue());
    }

    // Map squeeze dimensions to their positions after transpose.
    llvm::sort(squeeze_dims);
    llvm::SmallVector<int32_t, 4> new_squeeze_dims;
    for (int32_t dim : squeeze_dims) {
      new_squeeze_dims.push_back(perm[dim]);
    }
    llvm::sort(new_squeeze_dims);

    // Filter the original transpose permutation to keep only non-squeezed
    // positions.
    llvm::SmallVector<int32_t> filtered_perm_original_indices;
    for (int i = 0; i < input_rank; ++i) {
      if (!llvm::is_contained(squeeze_dims, i)) {
        filtered_perm_original_indices.push_back(perm[i]);
      }
    }

    // Map the remaining original dimension indices to new 0-based indices after
    // squeeze.
    llvm::SmallVector<int32_t> original_remaining_dims;
    for (int i = 0; i < input_rank; ++i) {
      if (!llvm::is_contained(new_squeeze_dims, i)) {
        original_remaining_dims.push_back(i);
      }
    }

    llvm::SmallVector<int32_t> original_to_new_index_map(input_rank, -1);
    for (int i = 0; i < original_remaining_dims.size(); ++i) {
      original_to_new_index_map[original_remaining_dims[i]] = i;
    }

    llvm::SmallVector<int32_t> new_perm;
    for (const auto &original_dim : filtered_perm_original_indices) {
      new_perm.push_back(original_to_new_index_map[original_dim]);
    }

    llvm::SmallVector<int64_t> new_squeeze_shape;
    for (int i = 0; i < input_rank; ++i) {
      if (!llvm::is_contained(new_squeeze_dims, i)) {
        new_squeeze_shape.push_back(
            transpose.getInput().getType().getDimSize(i));
      }
    }
    auto new_squeeze = rewriter.create<TFL::SqueezeOp>(
        squeeze->getLoc(),
        mlir::RankedTensorType::get(new_squeeze_shape,
                                    squeeze.getType().getElementType()),
        transpose.getInput(), rewriter.getI32ArrayAttr(new_squeeze_dims));

    auto new_transpose = rewriter.create<TFL::TransposeOp>(
        squeeze->getLoc(), squeeze.getType(), new_squeeze,
        rewriter.create<arith::ConstantOp>(
            squeeze->getLoc(), GetI32ElementsAttr(new_perm, &rewriter)));

    rewriter.replaceOp(squeeze, new_transpose);
    return success();
  }
};

// Helper function to check if a constant tensor attribute has the expected
// integer values
bool matchConstantIntPermutation(Value permValue,
                                 ArrayRef<int64_t> expectedPerm) {
  DenseElementsAttr permAttr;
  if (!matchPattern(permValue, m_Constant(&permAttr))) {
    return false;  // Not a constant
  }
  if (!permAttr.getElementType().isInteger(32) &&
      !permAttr.getElementType().isInteger(64)) {
    // TFLite perms are often i32, but accept i64 too
    return false;
  }

  auto values = permAttr.getValues<APInt>();
  if (values.size() != expectedPerm.size()) {
    return false;
  }
  for (size_t i = 0; i < expectedPerm.size(); ++i) {
    if (values[i].getSExtValue() != expectedPerm[i]) {
      return false;
    }
  }
  return true;
}

inline DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values,
                                               Builder *builder) {
  RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int32_t>(values.size())}, builder->getIntegerType(32));
  return DenseIntElementsAttr::get(ty, values);
}

inline DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int64_t> values,
                                               Builder *builder) {
  llvm::SmallVector<int32_t> new_values;
  for (auto el : values) {
    new_values.push_back(static_cast<int32_t>(el));
  }
  RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int32_t>(values.size())}, builder->getIntegerType(32));
  return DenseIntElementsAttr::get(ty, new_values);
}

// Reorders a Transpose-Reshape-Transpose sequence to
// Reshape-Transpose-Transpose to allow for further optimization.
//
// The pattern matches:
//   Transpose(Reshape(Transpose(input, perm: [1, 0])))
//
// and rewrites it to:
//   Transpose(Transpose(Reshape(input)))
//
// This reordering allows for further optimization by potentially fusing the
// reshapes and transposes.
struct ReorderTransposeReshapeTranspose
    : public OpRewritePattern<TFL::TransposeOp> {
  explicit ReorderTransposeReshapeTranspose(MLIRContext *context)
      : OpRewritePattern<TFL::TransposeOp>(context, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(TFL::TransposeOp outer_tpose,
                                PatternRewriter &rewriter) const override {
    auto reshape = outer_tpose.getInput().getDefiningOp<TFL::ReshapeOp>();
    if (!reshape) return failure();

    auto inner_tpose = reshape.getInput().getDefiningOp<TFL::TransposeOp>();
    if (!inner_tpose) return failure();

    auto inner_tpose_shape =
        mlir::dyn_cast_or_null<RankedTensorType>(inner_tpose.getType());
    if (!inner_tpose_shape) return failure();

    auto input = inner_tpose.getInput();

    auto inner_perm = inner_tpose.getPerm();
    if (!matchConstantIntPermutation(inner_perm, {1, 0})) return failure();

    int64_t perm0 = inner_tpose_shape.getDimSize(0);

    llvm::SmallVector<int32_t, 4> reshape_shape;
    {
      DenseIntElementsAttr reshape_shape_attr;
      if (!matchPattern(reshape.getShape(), m_Constant(&reshape_shape_attr))) {
        return failure();
      }

      for (auto dim : reshape_shape_attr) {
        reshape_shape.push_back(static_cast<int32_t>(dim.getSExtValue()));
      }
    }

    // Consume dimensions until we've equaled the size of the first dim in the
    // permuted result of the inner tpose and record the dim.
    int32_t dim = -1;
    for (auto i = 0, running_total = 1; i < reshape_shape.size(); i++) {
      running_total *= reshape_shape[i];
      if (perm0 == running_total) {
        dim = i;
      }
    }

    if (dim == -1) return failure();

    llvm::SmallVector<int64_t, 4> new_reshape_shape(reshape_shape.size());
    llvm::SmallVector<int32_t, 4> new_inner_perm(reshape_shape.size());

    int index = 0;
    for (auto i = dim + 1; i < reshape_shape.size(); i++) {
      new_inner_perm[i] = index;
      new_reshape_shape[index++] = reshape_shape[i];
    }
    for (auto i = 0; i <= dim; i++) {
      new_inner_perm[i] = index;
      new_reshape_shape[index++] = reshape_shape[i];
    }

    auto reshape_type =
        mlir::dyn_cast_or_null<RankedTensorType>(reshape.getType());
    if (!reshape_type) return failure();

    auto new_reshape_shape_const = rewriter.create<arith::ConstantOp>(
        reshape.getLoc(), GetI32ElementsAttr(new_reshape_shape, &rewriter));

    auto new_inner_reshape = rewriter.create<TFL::ReshapeOp>(
        reshape.getLoc(),
        RankedTensorType::get(new_reshape_shape, reshape_type.getElementType()),
        input, new_reshape_shape_const.getResult());
    auto new_inner_tpose = rewriter.create<TFL::TransposeOp>(
        inner_tpose.getLoc(), reshape_type, new_inner_reshape,
        rewriter.create<arith::ConstantOp>(
            inner_tpose.getLoc(),
            GetI32ElementsAttr(new_inner_perm, &rewriter)));

    rewriter.replaceOp(reshape, new_inner_tpose);

    return success();
  }
};

// Some models produce FullyConnected ops where the LHS is a const and the RHS
// is the activation. This breaks some downstream optimizations (notably input
// caching in XNNPack among other things). This rewrite pattern swaps the
// operands to match the expected order and recomputes a new output shape for
// the resuling op.
//
// This pattern only applies when:
// * input and filter operands are 2D
// * bias = none
// * keep_num_dims = false (implied if input and filter are 2D)
// Support for additional cases to broaden applicability can be added later.
// TODO(b/408313959): Add support for more cases.
//
// Note that transposes are added to maintain correctness:
//
// Original: Output[B, O] = FC(Input[B, I](Const), Filter[O, I](Var), Bias=None)
//                           ~= matmul(C, transpose(V))
//
// Transformed:
//   Intermediate[O, B] = FC(Filter[O, I](Var), Input[B, I](Const), None)
//                           ~= matmul(V, transpose(C))
//   FinalOutput[B, O]   = Transpose(Intermediate[O, B], perm=[1, 0])
struct FullyConnectedSwapOperandsWhenLHSIsConst
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  explicit FullyConnectedSwapOperandsWhenLHSIsConst(MLIRContext *context)
      : OpRewritePattern<TFL::FullyConnectedOp>(context, /*benefit=*/0) {}

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc,
                                PatternRewriter &rewriter) const override {
    if (!mlir::isa<NoneType>(fc.getBias().getType())) return failure();

    auto input = fc.getInput();
    auto filter = fc.getFilter();

    if (!matchPattern(input, m_Constant()) ||
        matchPattern(filter, m_Constant()))
      return failure();

    auto input_type = mlir::dyn_cast<RankedTensorType>(input.getType());
    auto filter_type = mlir::dyn_cast<RankedTensorType>(filter.getType());
    auto output_type =
        mlir::dyn_cast<RankedTensorType>(fc.getResult(0).getType());

    if (!input_type || !filter_type || !output_type) return failure();

    if (input_type.getRank() != 2 || filter_type.getRank() != 2)
      return failure();

    // Dimensions: B=Batch, I=InputDepth, O=OutputDepth
    // Input: [B, I], Filter: [O, I]
    // We extract B from the input operand and O from the filter operand
    int64_t B = input_type.getDimSize(0);
    int64_t O = filter_type.getDimSize(0);

    Type element_type = output_type.getElementType();
    Location loc = fc.getLoc();

    RankedTensorType intermediate_type =
        RankedTensorType::get({O, B}, element_type);

    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        loc,
        /*resultTypes=*/intermediate_type,
        /*input=*/filter,  // Original Filter V[O, I]
        /*filter=*/input,  // Original Input C[B, I]
        /*bias=*/fc.getBias(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc.getFusedActivationFunction()),
        /*weights_format=*/fc.getWeightsFormatAttr(),
        /*keep_num_dims=*/rewriter.getBoolAttr(false),
        /*asymmetric_quantize_inputs=*/
        fc.getAsymmetricQuantizeInputsAttr()  // Propagate quant attr
    );

    RankedTensorType final_shape_type =
        RankedTensorType::get({B, O}, element_type);

    Value transposed_result = rewriter.create<TFL::TransposeOp>(
        loc, final_shape_type, new_fc.getResult(0),
        rewriter.create<arith::ConstantOp>(
            loc, GetI32ElementsAttr(ArrayRef<int32_t>({1, 0}), &rewriter)));

    rewriter.replaceOp(fc, transposed_result);

    return success();
  }
};

// Adds canonicalization patterns to the list of patterns.
void AddCanonicalizationPatterns(MLIRContext *context,
                                 RewritePatternSet *patterns) {
  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}
}  // namespace

void OptimizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto *ctx = &getContext();
  auto func = getOperation();

  // Merge reshapes into fully connected ops before we start moving them past
  // binary ops.
  RewritePatternSet phase_0_patterns(&getContext());
  phase_0_patterns
      .add<SqueezeReshapesAroundBroadcastOp, RemoveReshapeAfterFullyConnected,
           RemoveReshapeBeforeFullyConnected,
           FuseOutputReshape_BatchMatMulWithFlattenedContractingDims,
           FuseSqueezingLhsReshapeIntoFC_Output,
           FuseReshapesAroundBatchMatMulLHS, FuseReshapesAroundBatchMatMulLHS1,
           FuseInputReshape_BatchMatMulWithFlattenedRhsDims,
           PushTransposeThroughSqueeze>(ctx);
  (void)applyPatternsGreedily(func, std::move(phase_0_patterns));

  // Potentially the binary ops might be fused together, like hard_swish, thus
  // we explore these potentially first and then fuse the binary ops with the
  // following ops in a second pattern match.
  TFL::populateWithGenerated(patterns);
  patterns
      .add<Convert2DUpscalingToResizeNearestNeighor, FuseFullyConnectedAndAdd,
           FuseAddAndFullyConnected, FuseFullyConnectedAndMul,
           FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
           FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
           FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>>(ctx);
  if (!GetOptions().disable_fuse_mul_and_fc) {
    patterns.add<FuseMulAndFullyConnected>(ctx);
  }
  if (GetOptions().enable_canonicalization) {
    AddCanonicalizationPatterns(ctx, &patterns);
  }
  (void)applyPatternsGreedily(func, std::move(patterns));

  // Fuse the binary ops with the following ops.
  RewritePatternSet phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.add<
      UndoBroadcastFullyConnectedBiasAddWithQDQs, FuseLogSoftmax,
      FuseFullyConnectedAndAdd, FuseAddAndFullyConnected,
      FuseFullyConnectedAndMul, FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
      FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
      FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>,
      FuseBinaryOpToFollowingConv2D, FuseBinaryOpToFollowingDepthwiseConv2D,
      FuseBinaryOpToFollowingFullyConnected, FuseConv2DAndMulWithQDQs,
      FuseDepthwiseConv2DAndMulWithQDQs, RemoveReshapeAfterFullyConnected,
      RemoveReshapeBeforeFullyConnected, FuseUnpackAndConcatToReshape,
      OptimizeTopK, FuseAddAndStridedSlice,
      FuseReshapeAndTransposeAroundBatchMatmul,
      FuseTransposeReshapeIntoBatchMatmul, MoveReshapeAfterFullyConnected,
      EnableFullyConnectedKeepNumDimsBeforeReshape,
      ReorderTransposeReshapeTranspose,
      FullyConnectedSwapOperandsWhenLHSIsConst>(ctx);
  if (!GetOptions().disable_fuse_mul_and_fc) {
    phase_2_patterns.add<FuseMulAndFullyConnected>(ctx);
  }
  if (GetOptions().enable_canonicalization) {
    AddCanonicalizationPatterns(ctx, &phase_2_patterns);
  }
  if (!GetOptions().enable_strict_qdq_mode) {
    phase_2_patterns.add<EliminateQDQPairs>(ctx);
  }
  (void)applyPatternsGreedily(func, std::move(phase_2_patterns));
}

}  // namespace TFL
}  // namespace mlir
