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

#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/ir/FakeQuantSupport.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantizeUtils.h"
#include "tensorflow/compiler/mlir/lite/quantization/ir/UniformSupport.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

namespace mlir {

// This includes the interface class definition. It couldn't be in a namespace
// because the table gen doesn't emit the namespace when it is used.
#include "tensorflow/compiler/mlir/lite/quantization/quantization_interface.cc.inc"

namespace quant {

namespace {
constexpr double kSmallestHalfRange = kNearZeroTolerance / 2;
using QType = quant::QuantizedType;

// This method expands the range to be larger than or equal to 1.0e-6, if it is
// very small (< 1.0e-6). This is to prevent very large quantized value by this
// range.
void ExpandVerySmallRange(ArrayRef<double> mins, ArrayRef<double> maxs,
                          SmallVectorImpl<double>* effective_mins,
                          SmallVectorImpl<double>* effective_maxs) {
  for (auto arg : llvm::zip(mins, maxs)) {
    double min = std::get<0>(arg);
    double max = std::get<1>(arg);
    // The range is wide, then use the same min/max.
    if ((max - min) > kNearZeroTolerance) {
      effective_mins->push_back(min);
      effective_maxs->push_back(max);
      continue;
    }

    // The range is small. Expands the range to stride 0.0 and also at least
    // 1.0e-6.
    effective_mins->push_back(std::min(min, -kSmallestHalfRange));
    effective_maxs->push_back(std::max(max, kSmallestHalfRange));
  }
}

// Set the min / max, scale and zero_points from the fake quant num_bits
// attribute from QAT.
QuantizedType ResetMinMaxFromNumBits(QuantizedType type, int num_bits,
                                     bool narrow_range, bool is_signed) {
  if (num_bits >= 8) {
    return type;
  }
  int64_t qmin = QType::getDefaultMinimumForInteger(is_signed, num_bits);
  int64_t qmax = QType::getDefaultMaximumForInteger(is_signed, num_bits);
  if (narrow_range) {
    qmin += 1;
  }
  const int64_t storage_type_min = type.getStorageTypeMin();
  const int64_t storage_type_max = type.getStorageTypeMax();
  const double rate =
      static_cast<double>(storage_type_max - storage_type_min) / (qmax - qmin);
  const auto& recalculate_scale = [&](double scale) -> double {
    return scale * rate;
  };
  const auto& recalculate_zero_point = [&](int64_t zero_point) -> int64_t {
    return qmax - std::round((storage_type_max - zero_point) / rate);
  };
  if (auto q_type = type.dyn_cast<UniformQuantizedType>()) {
    const double scale = recalculate_scale(q_type.getScale());
    const double zero_point = recalculate_zero_point(q_type.getZeroPoint());
    return UniformQuantizedType::get(q_type.getFlags(), q_type.getStorageType(),
                                     q_type.getExpressedType(), scale,
                                     zero_point, qmin, qmax);
  } else if (auto q_type =
                 type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    const int size = q_type.getScales().size();
    SmallVector<double, 4> scales(size);
    SmallVector<int64_t, 4> zero_points(size);
    for (int i = 0; i < size; ++i) {
      scales[i] = recalculate_scale(q_type.getScales()[i]);
      zero_points[i] = recalculate_zero_point(q_type.getZeroPoints()[i]);
    }
    return quant::UniformQuantizedPerAxisType::get(
        q_type.getFlags(), q_type.getStorageType(), q_type.getExpressedType(),
        scales, zero_points, q_type.getQuantizedDimension(), qmin, qmax);
  } else {
    llvm_unreachable("Unsupported QuantizedType in ResetMinMaxFromNumBits");
  }
  return type;
}

// Repeats the content of `data` multiple times to resize to `target_size`.
// Note that this only broadcast across one dimension.
template <typename T>
bool BroadcastVector(int target_size, SmallVectorImpl<T>& data) {
  int size = data.size();
  if (size != target_size) {
    if (target_size % size != 0) return true;
    data.reserve(target_size);
    for (int i = 1, e = target_size / size; i != e; ++i) {
      data.insert(data.end(), data.begin(), data.begin() + size);
    }
  }
  return false;
}

// Changes the axis of the input per-channel quantized type to match the
// dimension of the target type. Returns nullptr if it fails.
quant::UniformQuantizedPerAxisType ResetAxisAndBroadcast(
    ArrayRef<int64_t> shape, quant::UniformQuantizedPerAxisType qtype,
    Type target, int quant_dim) {
  auto shaped = target.dyn_cast<RankedTensorType>();
  if (!shaped) return {};
  ArrayRef<int64_t> new_shape = shaped.getShape();

  SmallVector<double, 4> scales(qtype.getScales().begin(),
                                qtype.getScales().end());
  SmallVector<int64_t, 4> zero_points(qtype.getZeroPoints().begin(),
                                      qtype.getZeroPoints().end());

  if (new_shape.size() == shape.size()) {  // same rank
    // Broadcast the scales and zero points to match the target size, which is
    // usually the axis-th dimension of the target type. Currently, it covers
    // two cases:
    // - for Transpose, the data layout is changed so the `dim[axis]` still
    // equals to the `scales_size`. The broadcast skips;
    // - for Reshape, the data layout isn't changed but the innermost dimension
    // is expand to cover the last two original dimensions. Thus we just need to
    // be repeated the `scales` dim[2] times to covers the new dim length.
    //
    // TODO(b/141709944): after the fix, the `scales` can be for dim[2], thus we
    // have to repeat each elements in the `scales` locally dim[3] times.
    if (BroadcastVector<double>(shaped.getDimSize(quant_dim), scales) ||
        BroadcastVector<int64_t>(shaped.getDimSize(quant_dim), zero_points)) {
      return {};
    }
  } else if ((new_shape.size() == shape.size() + 1) && new_shape.front() == 1) {
    // Handle the [A, B, C] -> [1, A, B, C] reshape case.
    if (!(std::equal(shape.begin(), shape.end(), new_shape.begin() + 1) &&
          quant_dim == new_shape.size() - 1)) {
      return {};
    }
  } else {
    return {};
  }

  return quant::UniformQuantizedPerAxisType::get(
      qtype.getFlags(), qtype.getStorageType(), qtype.getExpressedType(),
      scales, zero_points, quant_dim, qtype.getStorageTypeMin(),
      qtype.getStorageTypeMax());
}

}  // namespace

bool IsOpNotQuantizable(Operation* op) {
  // If it is terminator or not quantizable or any ops form the mlir quant
  // ops dialect, we shouldn't rewrite.
  bool attr_enforced_quantizable =
      op->hasAttrOfType<StringAttr>(kQuantTraitAttrName) &&
      op->getAttrOfType<StringAttr>(kQuantTraitAttrName).getValue().str() ==
          QuantTraitValues[QuantizationTrait::FullyQuantizable];

  // Constant ops do not have QuantizableResult attribute but they can deal with
  // quantized tensors.
  if (llvm::isa<func::ConstantOp, arith::ConstantOp, quantfork::StatisticsOp>(
          op))
    return false;

  bool prop_enforced_quantizable =
      op->hasTrait<OpTrait::quant::QuantizableResult>();

  return op->hasTrait<OpTrait::IsTerminator>() ||
         llvm::isa<quantfork::QuantizeCastOp, quantfork::DequantizeCastOp>(
             op) ||
         (!attr_enforced_quantizable && !prop_enforced_quantizable);
}

// Returns the quantized type for the
// input_type/min/max/storag_type_width/narrow_range.
// This is entry point to the Quant dialect and used for both quantizing
// activations and weights.
Type GetQuantizedType(Builder builder, Type input_type, ArrayRef<double> min,
                      ArrayRef<double> max, int quant_dim,
                      int storage_type_width, bool narrow_range, bool is_signed,
                      bool legacy_float_scale, bool use_fake_quant_num_bits) {
  auto converter =
      quantfork::ExpressedToQuantizedConverter::forInputType(input_type);

  // Expand the range to prevent extremely small scales and large quantized
  // integers which can cause overflow. This leads to scale
  // 7.843137254901961e-9 with 8 bits.
  SmallVector<double, 4> effective_mins, effective_maxs;
  ExpandVerySmallRange(min, max, &effective_mins, &effective_maxs);

  quant::QuantizedType quantizedEleType;
  if (min.size() == 1 && max.size() == 1 && quant_dim == -1) {
    quantizedEleType = quantfork::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, effective_mins[0],
        effective_maxs[0], narrow_range, converter.expressedType, is_signed);
    if (legacy_float_scale) {
      quantizedEleType =
          DownCastScale(quantizedEleType, effective_mins[0], effective_maxs[0],
                        builder.getUnknownLoc());
    }
  } else if (min.size() == max.size()) {
    auto shape = input_type.dyn_cast<ShapedType>();
    if (!shape || shape.getRank() <= quant_dim ||
        static_cast<int64_t>(min.size()) != shape.getDimSize(quant_dim)) {
      return {};
    }
    // The quantization dim is set to the last dimension.
    quantizedEleType = quantfork::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, quant_dim, effective_mins,
        effective_maxs, narrow_range, converter.expressedType, is_signed);
    if (legacy_float_scale) {
      quantizedEleType = DownCastScale(quantizedEleType, effective_mins,
                                       effective_maxs, builder.getUnknownLoc());
    }
  }
  if (!quantizedEleType) return {};
  // Use fake quant configured bit-widths (only supported for
  // 1 < num_bits < 8 bits) instead of using 8bit defaults.
  if (use_fake_quant_num_bits && (storage_type_width > 1) &&
      (storage_type_width < 8) &&
      (quantizedEleType.getStorageTypeMax() >
       QType::getDefaultMinimumForInteger(is_signed, storage_type_width))) {
    auto resetEleType = ResetMinMaxFromNumBits(
        quantizedEleType, storage_type_width, narrow_range, is_signed);
    return converter.convert(resetEleType);
  }
  return converter.convert(quantizedEleType);
}

// TODO(fengliuai): promote this utility method to mlir QuantOps.
TypeAttr RescaleQuantizedType(Type input, Attribute factor) {
  auto factor_values = factor.dyn_cast_or_null<DenseFPElementsAttr>();
  if (!factor_values) return {};
  auto ele_type = quant::QuantizedType::getQuantizedElementType(input);
  if (!ele_type) return {};
  if (auto qtype = ele_type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    ArrayRef<double> scales = qtype.getScales();
    // Broadcasting hasn't been implemented yet.
    if (static_cast<int64_t>(scales.size()) != factor_values.getNumElements())
      return {};
    SmallVector<double, 4> new_scales;
    new_scales.reserve(scales.size());
    auto scales_iter = scales.begin();
    for (const auto& f : factor_values) {
      new_scales.push_back(*(scales_iter++) *
                           std::fabs(FloatAttr::getValueAsDouble(f)));
    }
    // We are assuming symmetric quantization.
    auto new_ele_type = quant::UniformQuantizedPerAxisType::get(
        qtype.getFlags(), qtype.getStorageType(), qtype.getExpressedType(),
        new_scales, qtype.getZeroPoints(), qtype.getQuantizedDimension(),
        qtype.getStorageTypeMin(), qtype.getStorageTypeMax());
    if (auto new_type = new_ele_type.castFromExpressedType(
            quant::QuantizedType::castToExpressedType(input))) {
      return TypeAttr::get(new_type);
    }
  }
  // Currently, we only support per-axis quantized type.
  return {};
}

TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, int quant_dim,
                              IntegerAttr num_bits, BoolAttr narrow_range,
                              bool is_signed, bool legacy_float_scale,
                              bool use_fake_quant_num_bits) {
  SmallVector<double, 4> min_value, max_value;
  auto mins = min.dyn_cast<DenseFPElementsAttr>();
  auto maxs = max.dyn_cast<DenseFPElementsAttr>();
  if (mins && maxs) {
    min_value.reserve(mins.getNumElements());
    max_value.reserve(maxs.getNumElements());
    for (auto it = mins.begin(), e = mins.end(); it != e; ++it) {
      min_value.push_back(FloatAttr::getValueAsDouble(*it));
    }
    for (auto it = maxs.begin(), e = maxs.end(); it != e; ++it) {
      max_value.push_back(FloatAttr::getValueAsDouble(*it));
    }
  } else {
    auto fmin = min.dyn_cast<FloatAttr>();
    auto fmax = max.dyn_cast<FloatAttr>();
    if (fmin && fmax) {
      min_value.push_back(fmin.getValueAsDouble());
      max_value.push_back(fmax.getValueAsDouble());
    } else {
      return {};
    }
  }
  Type final_type =
      GetQuantizedType(builder, input_type, min_value, max_value, quant_dim,
                       num_bits.getInt(), narrow_range.getValue(), is_signed,
                       legacy_float_scale, use_fake_quant_num_bits);
  if (!final_type) return {};
  return TypeAttr::get(final_type);
}

TypeAttr CastQuantizedTypeAttrFromExpressedType(Builder builder,
                                                TypeAttr source, Type target,
                                                int axis) {
  auto source_type = source.getValue().dyn_cast_or_null<ShapedType>();
  if (!source_type) return {};
  auto src_ele_type = source_type.getElementType();
  auto qtype = src_ele_type.dyn_cast<quant::QuantizedType>();

  // Reset the quantization dimensions if it is per-axis.
  if (auto per_axis =
          qtype.dyn_cast_or_null<quant::UniformQuantizedPerAxisType>()) {
    // For the pass-through ops, we don't know which the dimension will be the
    // new quantization dimension. Only if the new quantization dimension can
    // be inferred, it is safe to reset the per-axis quantized type.
    if (axis == -1) return {};
    qtype =
        ResetAxisAndBroadcast(source_type.getShape(), per_axis, target, axis);
  }
  if (!qtype) return {};
  Type final_type = qtype.castFromExpressedType(target);
  if (!final_type) return {};
  return TypeAttr::get(final_type);
}

void ExtractMinMaxFromAttr(DenseFPElementsAttr values, int dim_size,
                           int slice_size, bool symmetric,
                           SmallVectorImpl<double>& mins,
                           SmallVectorImpl<double>& maxs) {
  // If all the element values are same we don't need to scan the content.
  if (values.isSplat()) {
    double single_value =
        FloatAttr::getValueAsDouble(values.getSplatValue<llvm::APFloat>());

    // When the single value isn't 0.0, we expand it to a range to include
    // this single value and 0.0. This will give us a scale and zero point
    // works for both this value and 0.0.
    if (single_value < 0.0) {
      mins[0] = single_value;
      maxs[0] = symmetric ? -single_value : 0.0;
    } else if (single_value > 0.0) {
      mins[0] = symmetric ? -single_value : 0.0;
      maxs[0] = single_value;
    } else {
      mins[0] = maxs[0] = single_value;
    }
    for (int i = 1; i < dim_size; ++i) {
      mins[i] = mins[0];
      maxs[i] = maxs[0];
    }
  } else {
    int64_t flatten_index = 0;
    for (auto it = values.begin(), e = values.end(); it != e;
         ++it, ++flatten_index) {
      double ele_value = FloatAttr::getValueAsDouble(*it);
      int slice_index = flatten_index / slice_size;
      int channel_index = slice_index % dim_size;
      mins[channel_index] = std::min(mins[channel_index], ele_value);
      maxs[channel_index] = std::max(maxs[channel_index], ele_value);
    }
    // Expand range to include 0.
    for (int i = 0; i < dim_size; ++i) {
      maxs[i] = std::max(maxs[i], 0.0);
      mins[i] = std::min(mins[i], 0.0);
    }
    if (symmetric) {
      for (int i = 0; i < dim_size; ++i) {
        maxs[i] = std::max(std::abs(mins[i]), std::abs(maxs[i]));
        mins[i] = -maxs[i];
      }
    }
  }
}

Type GetUniformQuantizedTypeForWeight(ElementsAttr attr, bool symmetric,
                                      unsigned num_bits, bool is_signed,
                                      bool narrow_range,
                                      bool legacy_float_scale,
                                      bool use_fake_quant_num_bits) {
  Builder builder(attr.getContext());
  // `symmetric` can only be used when it is `signed` and `narrow_range`.
  if (symmetric && (!is_signed || !narrow_range)) return {};

  SmallVector<double, 4> mins(1, std::numeric_limits<double>::max());
  SmallVector<double, 4> maxs(1, std::numeric_limits<double>::min());
  auto fp = attr.dyn_cast<DenseFPElementsAttr>();
  if (!fp) return {};

  // Computes the effective min/max values of the attribute values.
  ExtractMinMaxFromAttr(fp, /*dim_size=*/1, /*slice_size=*/1, symmetric, mins,
                        maxs);

  auto type =
      GetQuantizedType(builder, attr.getType(), mins[0], maxs[0],
                       /*quant_dim=*/-1, num_bits, narrow_range, is_signed,
                       legacy_float_scale, use_fake_quant_num_bits);
  if (auto ele_type = type.dyn_cast_or_null<TensorType>())
    return ele_type.getElementType();

  return {};
}

Type GetUniformQuantizedPerAxisTypeForWeight(ElementsAttr attr, int quant_dim,
                                             bool symmetric, unsigned num_bits,
                                             bool is_signed, bool narrow_range,
                                             bool legacy_float_scale,
                                             bool use_fake_quant_num_bits) {
  Builder builder(attr.getContext());
  auto shape = attr.getType().cast<ShapedType>().getShape();
  if (static_cast<int>(shape.size()) <= quant_dim) return {};
  // `symmetric` can only be used when it is `signed` and `narrow_range`.
  if (symmetric && (!is_signed || !narrow_range)) return {};

  int dim_size = shape[quant_dim];
  int slice_size = std::accumulate(std::next(shape.begin(), quant_dim + 1),
                                   shape.end(), 1, std::multiplies<int64_t>());
  SmallVector<double, 4> mins(dim_size, std::numeric_limits<double>::max());
  SmallVector<double, 4> maxs(dim_size, std::numeric_limits<double>::min());
  auto fp = attr.dyn_cast<DenseFPElementsAttr>();
  if (!fp) return {};

  // Computes the effective min/max values of the attribute values.
  ExtractMinMaxFromAttr(fp, dim_size, slice_size, symmetric, mins, maxs);

  auto type = GetQuantizedType(builder, attr.getType(), mins, maxs, quant_dim,
                               num_bits, narrow_range, is_signed,
                               legacy_float_scale, use_fake_quant_num_bits);
  if (auto ele_type = type.dyn_cast_or_null<TensorType>())
    return ele_type.getElementType();

  return {};
}

quant::QuantizedType GetUniformQuantizedTypeForBias(
    const std::vector<quant::QuantizedType>& op_types,
    bool legacy_float_scale) {
  if (op_types.empty()) return {};

  size_t axis_size = 1;
  int32_t quant_dim = -1;
  Type expressed_type;
  // Requires all the op types are valid UniformQuantizedTypes or
  // UniformQuantizedPerAxisTypes and also have same expressed type. For all
  // the UniformQuantizedPerAxisTypes, the quantization dimension index and
  // dimension sizes are same.
  for (auto op_type : op_types) {
    if (!op_type) return {};
    if (expressed_type && expressed_type != op_type.getExpressedType()) {
      return {};
    }
    expressed_type = op_type.getExpressedType();

    if (auto type = op_type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
      if ((axis_size != 1 && axis_size != type.getScales().size())) return {};
      if (quant_dim != -1 && quant_dim != type.getQuantizedDimension())
        return {};
      axis_size = type.getScales().size();
      quant_dim = type.getQuantizedDimension();
    } else if (!op_type.isa<quant::UniformQuantizedType>()) {
      return {};
    }
  }

  // The scale from the UniformQuantizedTypes is broadcasted if there are
  // UniformQuantizedPerAxisTypes.
  llvm::SmallVector<double, 4> scales(axis_size, 1.0);
  for (auto op_type : op_types) {
    if (auto type = op_type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
      for (const auto& index_scale : llvm::enumerate(type.getScales())) {
        scales[index_scale.index()] *= index_scale.value();
      }
    } else if (auto type = op_type.dyn_cast<quant::UniformQuantizedType>()) {
      for (int index = 0, e = axis_size; index != e; ++index) {
        scales[index] *= type.getScale();
      }
    }
  }
  if (legacy_float_scale) {
    for (int i = 0; i < scales.size(); ++i) {
      scales[i] = static_cast<float>(scales[i]);
    }
  }

  // Builds the result quantized type, which has signed 32 bits storage type.
  Builder builder(expressed_type.getContext());
  IntegerType storage_type = builder.getIntegerType(32);
  int64_t storage_type_min =
      quant::QuantizedType::getDefaultMinimumForInteger(/*isSigned=*/true, 32);
  int64_t storage_type_max =
      quant::QuantizedType::getDefaultMaximumForInteger(/*isSigned=*/true, 32);
  if (axis_size == 1) {
    return quant::UniformQuantizedType::getChecked(
        builder.getUnknownLoc(),
        /*flags=*/true, storage_type, expressed_type, scales[0],
        /*zeroPoint=*/0, storage_type_min, storage_type_max);
  } else {
    llvm::SmallVector<int64_t, 4> zero_points(axis_size, 0);
    // Assume the bias is a 1-D tensor, and set the quantization dim to the last
    // dimension, which is 0. If the bias rank is larger than 1, this returned
    // quantized type couldn't be used to quantize the bias.
    return quant::UniformQuantizedPerAxisType::getChecked(
        builder.getUnknownLoc(),
        /*flags=*/true, storage_type, expressed_type, scales, zero_points,
        /*quantizedDimension=*/0, storage_type_min, storage_type_max);
  }
}

ElementsAttr QuantizeLegacy(Attribute real_value, Type tensor_type) {
  if (!real_value.isa<DenseFPElementsAttr>() ||
      !quant::QuantizedType::getQuantizedElementType(tensor_type)) {
    return {};
  }
  auto real_values_attr = real_value.cast<DenseFPElementsAttr>();
  auto q_type = quant::QuantizedType::getQuantizedElementType(tensor_type);
  std::vector<float> real_values;
  llvm::SmallVector<APInt, 8> quantized_attr;
  real_values.reserve(real_values_attr.getNumElements());
  quantized_attr.reserve(real_values_attr.getNumElements());
  std::transform(real_values_attr.begin(), real_values_attr.end(),
                 std::back_inserter(real_values), [&](APFloat value) -> float {
                   return value.convertToFloat();
                 });
  ShapedType new_dense_type =
      q_type.castExpressedToStorageType(real_values_attr.getType())
          .dyn_cast_or_null<ShapedType>();
  int width = q_type.getStorageType().dyn_cast<mlir::IntegerType>().getWidth();

  if (width == 8 && q_type.getStorageTypeMax() == 127 &&
      q_type.getStorageTypeMin() == -127) {
    std::vector<int8_t> quantized_values(real_values_attr.getNumElements());
    if (auto uniform_type = q_type.dyn_cast<UniformQuantizedType>()) {
      float min, max, scale;
      tflite::tensor_utils::SymmetricQuantizeFloats(
          real_values.data(), real_values.size(), quantized_values.data(), &min,
          &max, &scale);
      // The scale has been adjusted, so the adjusted scale should be respected.
      if (std::abs(scale - uniform_type.getScale()) > 1e-3) {
        return Quantize(real_value, tensor_type);
      }
    } else if (auto uniform_type =
                   q_type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
      std::vector<float> scales_inv;
      std::vector<int32_t> dimension;
      dimension.insert(dimension.end(), new_dense_type.getShape().begin(),
                       new_dense_type.getShape().end());
      std::transform(uniform_type.getScales().begin(),
                     uniform_type.getScales().end(),
                     std::back_inserter(scales_inv),
                     [](float scale) { return 1.0 / scale; });

      tflite::optimize::utils::SymmetricPerChannelQuantizeValues(
          real_values.data(), scales_inv, dimension,
          uniform_type.getQuantizedDimension(), &quantized_values);
    } else {
      return {};
    }
    std::transform(quantized_values.begin(), quantized_values.end(),
                   std::back_inserter(quantized_attr),
                   [&](int8_t value) -> APInt {
                     return APInt(8, value, /*isSigned=*/true);
                   });
    return DenseElementsAttr::get(new_dense_type, quantized_attr);
  } else if (width == 8) {
    // This can be a state tensor, or an actual constant tensor with
    // asymmetric range. For a state tensor, assigining correct quantization
    // parameters is sufficient, and for constants with asymmetric range it's
    // not correctly quantized by legacy quantizer so call the new Quantize.
    return Quantize(real_value, tensor_type);
  } else if (width == 16) {
    if (auto uniform_type = q_type.dyn_cast<UniformQuantizedType>()) {
      auto quantized_values =
          tflite::optimize::utils::SymmetricQuantizeFloatsToInt16(
              real_values.data(), real_values.size(), uniform_type.getScale());
      std::transform(quantized_values.begin(), quantized_values.end(),
                     std::back_inserter(quantized_attr),
                     [&](int16_t value) -> APInt {
                       return APInt(16, value, /*isSigned=*/true);
                     });
      return DenseElementsAttr::get(new_dense_type, quantized_attr);
    }
  } else if (width == 32) {
    std::vector<float> scales;
    if (auto uniform_type = q_type.dyn_cast<UniformQuantizedType>()) {
      scales.push_back(uniform_type.getScale());
    } else if (auto uniform_type =
                   q_type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
      scales.insert(scales.end(), uniform_type.getScales().begin(),
                    uniform_type.getScales().end());
    } else {
      return {};
    }
    auto quantized_bias =
        tflite::optimize::utils::SymmetricBiasQuantize<std::int32_t>(
            real_values.data(), real_values.size(), scales);
    std::transform(quantized_bias.begin(), quantized_bias.end(),
                   std::back_inserter(quantized_attr),
                   [&](int32_t value) -> APInt {
                     return APInt(32, value, /*isSigned=*/true);
                   });
    return DenseElementsAttr::get(new_dense_type, quantized_attr);
  }
  return {};
}

ElementsAttr Quantize(Attribute real_value, Type tensor_type) {
  if (auto q_type =
          quant::QuantizedType::getQuantizedElementType(tensor_type)) {
    Type converted_type;
    return quantfork::quantizeAttr(real_value, q_type, converted_type)
        .dyn_cast<ElementsAttr>();
  }
  return {};
}

quant::QuantizedType DownCastScale(QuantizedType type, double min, double max,
                                   Location loc) {
  SmallVector<double, 1> mins = {min};
  SmallVector<double, 1> maxs = {max};
  return DownCastScale(type, mins, maxs, loc);
}

quant::QuantizedType DownCastScale(QuantizedType type,
                                   const SmallVectorImpl<double>& mins,
                                   const SmallVectorImpl<double>& maxs,
                                   Location loc) {
  // The given type can be null. For example, there can be an invalid scale and
  // so on.
  if (!type) return type;
  SmallVector<double, 4> scales(mins.size());
  SmallVector<int64_t, 4> zero_points(mins.size());
  if (auto q_type = type.dyn_cast<UniformQuantizedType>()) {
    zero_points.push_back(q_type.getZeroPoint());
  } else if (auto q_type =
                 type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    zero_points = {q_type.getZeroPoints().begin(),
                   q_type.getZeroPoints().end()};
  }
  for (int i = 0; i < mins.size(); ++i) {
    scales[i] = (static_cast<float>(maxs[i]) - static_cast<float>(mins[i])) /
                (type.getStorageTypeMax() - type.getStorageTypeMin());
    if (type.getStorageTypeMax() != -type.getStorageTypeMin()) {
      // Only applies for asymmetric quantized range with original scale.
      float zero_point_from_min =
          type.getStorageTypeMin() - mins[i] / scales[i];
      if (zero_point_from_min < type.getStorageTypeMin()) {
        zero_points[i] = static_cast<int64_t>(type.getStorageTypeMin());
      } else if (zero_point_from_min > type.getStorageTypeMax()) {
        zero_points[i] = static_cast<int64_t>(type.getStorageTypeMax());
      } else {
        zero_points[i] = static_cast<int64_t>(std::round(zero_point_from_min));
      }
    }
  }
  if (auto q_type = type.dyn_cast<UniformQuantizedType>()) {
    return UniformQuantizedType::get(q_type.getFlags(), q_type.getStorageType(),
                                     q_type.getExpressedType(), scales[0],
                                     zero_points[0], q_type.getStorageTypeMin(),
                                     q_type.getStorageTypeMax());
  } else if (auto q_type =
                 type.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    return quant::UniformQuantizedPerAxisType::get(
        q_type.getFlags(), q_type.getStorageType(), q_type.getExpressedType(),
        scales, zero_points, q_type.getQuantizedDimension(),
        q_type.getStorageTypeMin(), q_type.getStorageTypeMax());
  }
  return type;
}

// A heuristic to determine whether the scales needs to be from operands or
// from results for the ops with the `SameOperandsAndResultsScale` property.
// The current implementation is based on the number of operands.
static bool PreferResultScale(Operation* op) {
  int float_operands = 0;
  for (auto operand : op->getOperands()) {
    if (auto operand_type = operand.getType().dyn_cast<ShapedType>()) {
      if (operand_type.getElementType().isa<FloatType>()) {
        if (++float_operands > 1) return true;
      }
    }
  }
  return false;
}

std::unique_ptr<OpQuantScaleSpec> GetDefaultQuantScaleSpec(Operation* op) {
  auto spec = std::make_unique<OpQuantScaleSpec>();
  if (llvm::isa<SameScalesOpInterface>(op)) {
    spec->has_same_scale_requirement = true;
    spec->required_same_scale_func = [op](bool sign, int bit_width) {
      return llvm::cast<SameScalesOpInterface>(op)
          .RequiredSameOperandsAndResultsScale(sign, bit_width);
    };
    spec->required_same_quantized_axes_func = [op]() {
      return llvm::cast<SameScalesOpInterface>(op).RequiredSameQuantizedAxes();
    };
  }
  if (llvm::isa<FixedOutputRangeInterface>(op)) {
    spec->has_fixed_output_range = true;
    spec->fixed_output_range_func = [op](bool sign, int bit_width) {
      return llvm::cast<FixedOutputRangeInterface>(op).GetFixedOutputRange(
          sign, bit_width);
    };
  }
  return spec;
}

// The stats op of some of the ops can be redundant. The current implementation
// only considers the ops with restricted output params.
static bool IsStatsRedundant(
    Operation* op, OpQuantSpecGetter op_quant_spec_getter,
    OpQuantScaleSpecGetter op_quant_scale_spec_getter) {
  // If it has FixedOutputRangeInterface, no need to manually create spec.
  return llvm::isa<FixedOutputRangeInterface>(op) ||
         op_quant_scale_spec_getter(op)->has_fixed_output_range;
}

static bool IsSameScaleOp(Operation* op,
                          OpQuantScaleSpecGetter op_quant_scale_spec_getter) {
  // If it has SameScalesOpInterface, no need to manually create spec.
  return llvm::dyn_cast<SameScalesOpInterface>(op) ||
         op_quant_scale_spec_getter(op)->has_same_scale_requirement;
}

bool RemoveRedundantStatsOps(
    mlir::func::FuncOp func, OpQuantSpecGetter op_quant_spec_getter,
    OpQuantScaleSpecGetter op_quant_scale_spec_getter) {
  llvm::SmallVector<quantfork::StatisticsOp, 16> all_stats_ops;
  llvm::DenseSet<Operation*> redundant_stats_ops;

  // Step 0: remove the quantfork::StatisticsOp which are used by the
  // quant.qcast op in case it overrides the information from training FakeQuant
  // ops.
  func.walk([&](quantfork::QuantizeCastOp q) {
    auto input_op = q.getArg().getDefiningOp();
    if (auto stats =
            llvm::dyn_cast_or_null<quantfork::StatisticsOp>(input_op)) {
      q.setOperand(stats.getArg());
      if (stats.use_empty()) stats.erase();
    }
  });

  // Step 1: forward pass: propagate any value scales which are not produces
  // by `SameOperandsAndResultsScale`. Additionally, remove the value scales
  // which are produced by the ops with the `FixedOutputRangeInterface`.
  // Note that we don't propagate across the multiple-operands
  // `SameOperandsAndResultsScale` ops like `concatenation`.
  func.walk([&](quantfork::StatisticsOp stats_op) {
    all_stats_ops.push_back(stats_op);
  });

  while (!all_stats_ops.empty()) {
    quantfork::StatisticsOp stats_op = all_stats_ops.back();
    all_stats_ops.pop_back();

    if (auto def = stats_op.getArg().getDefiningOp()) {
      if (IsStatsRedundant(def, op_quant_spec_getter,
                           op_quant_scale_spec_getter)) {
        redundant_stats_ops.insert(stats_op);
      }
    }

    for (auto user : stats_op.getResult().getUsers()) {
      // We don't propagate this parameter down if it has multiple operands.
      // We want to use the result parameter scales instead.
      if (!IsSameScaleOp(user, op_quant_scale_spec_getter) ||
          PreferResultScale(user)) {
        continue;
      }
      for (Value res : user->getResults()) {
        if (!res.hasOneUse()) {
          continue;
        }
        if (auto next_stats = llvm::dyn_cast<quantfork::StatisticsOp>(
                *res.getUsers().begin())) {
          // quantization parameters can be propagated to next_stats
          redundant_stats_ops.insert(next_stats);
          // add next_stats to the work list so propagation can continue.
          all_stats_ops.push_back(next_stats);
        }
      }
    }
  }

  // Step 2: backward pass: For the ops skiped in the forward pass, propagate
  // its results scale backwards as far as possible.
  func.walk([&](quantfork::StatisticsOp stats_op) {
    if (redundant_stats_ops.find(stats_op) == redundant_stats_ops.end()) {
      all_stats_ops.push_back(stats_op);
    }
  });

  while (!all_stats_ops.empty()) {
    quantfork::StatisticsOp stats_op = all_stats_ops.back();
    all_stats_ops.pop_back();

    if (auto def = stats_op.getArg().getDefiningOp()) {
      if (!IsSameScaleOp(def, op_quant_scale_spec_getter)) {
        continue;
      }
      for (auto input : def->getOperands()) {
        if (auto next_stats = llvm::dyn_cast_or_null<quantfork::StatisticsOp>(
                input.getDefiningOp())) {
          redundant_stats_ops.insert(next_stats);
          all_stats_ops.push_back(next_stats);
        }
      }
    }
  }

  // Step3: Remove all the redundant stats ops
  for (auto it : redundant_stats_ops) {
    if (!llvm::isa<quantfork::StatisticsOp>(it)) return true;
    auto stats_op = llvm::cast<quantfork::StatisticsOp>(it);
    stats_op.getResult().replaceAllUsesWith(stats_op.getArg());
    stats_op.erase();
  }

  // Returns false if the steps finish without errors.
  return false;
}

LogicalResult VerifySameScales(Operation* op) {
  auto same_scale_op = llvm::cast<SameScalesOpInterface>(op);

  llvm::SmallVector<QuantizedType, 4> collected_quant_params;
  for (auto input : op->getOperands()) {
    auto quant_params = QuantizedType::getQuantizedElementType(input.getType());
    // Skip non-quantizable operands.
    if (quant_params) {
      collected_quant_params.push_back(quant_params);
    }
  }

  for (auto output : op->getResults()) {
    auto quant_params =
        QuantizedType::getQuantizedElementType(output.getType());
    // Skip non-quantizable results.
    if (quant_params) {
      collected_quant_params.push_back(quant_params);
    }
  }

  if (collected_quant_params.size() <= 1) return success();
  const auto& expected_params = collected_quant_params[0];
  for (int i = 1; i < collected_quant_params.size(); i++) {
    const auto& compared_params = collected_quant_params[i];
    // For some ops (such as Transpose or Squeeze), the quantized axis might not
    // be the same, this function only verifies the scale and zero point in
    // that case. The quantized axis should be verified in their own verifier
    // method.
    if (!same_scale_op.RequiredSameQuantizedAxes()) {
      auto expected_per_axis_qtype =
          expected_params.dyn_cast<quant::UniformQuantizedPerAxisType>();
      auto compared_per_axis_qtype =
          compared_params.dyn_cast<quant::UniformQuantizedPerAxisType>();
      if (expected_per_axis_qtype && compared_per_axis_qtype &&
          llvm::equal(expected_per_axis_qtype.getScales(),
                      compared_per_axis_qtype.getScales()) &&
          llvm::equal(expected_per_axis_qtype.getZeroPoints(),
                      compared_per_axis_qtype.getZeroPoints()) &&
          expected_params.getStorageType() ==
              compared_params.getStorageType() &&
          expected_params.getExpressedType() ==
              compared_params.getExpressedType()) {
        continue;
      }
    }
    // Same quantization parameters are always ok.
    if (expected_params == compared_params) continue;
    // If the quantization parameters are not the same, as long as it has the
    // same storage type and the op interface doesn't require same scale
    // constraint for this storage type, it is still ok.
    if ((expected_params.isSigned() == compared_params.isSigned() &&
         expected_params.getStorageTypeIntegralWidth() ==
             compared_params.getStorageTypeIntegralWidth()) &&
        !same_scale_op.RequiredSameOperandsAndResultsScale(
            expected_params.isSigned(),
            expected_params.getStorageTypeIntegralWidth()))
      continue;

    std::string err_msg =
        "quantization parameters violate the same scale constraint: ";
    llvm::raw_string_ostream os(err_msg);
    expected_params.print(os);
    os << " vs. ";
    compared_params.print(os);
    os.flush();
    return op->emitOpError(err_msg);
  }
  return success();
}

quant::UniformQuantizedType GetFixedOutputRange(bool is_signed, int bit_width,
                                                Type tensor_type, double scale,
                                                int64_t zero_point,
                                                int64_t storage_min,
                                                int64_t storage_max) {
  auto result_type = tensor_type.cast<ShapedType>();
  if (!result_type.getElementType().isa<FloatType>()) return {};
  Builder builder(result_type.getContext());

  // Only support 8-bits
  if (bit_width != 8) return {};
  IntegerType storage_type = builder.getIntegerType(bit_width);
  if (!is_signed) {
    zero_point += 128;
    storage_min += 128;
    storage_max += 128;
  }
  return quant::UniformQuantizedType::getChecked(
      builder.getUnknownLoc(), is_signed, storage_type,
      result_type.getElementType(), scale, zero_point, storage_min,
      storage_max);
}

Type ConvertSignedQuantizedToUnsigned(Type signed_tensor_type, Location loc) {
  auto qtype = QType::getQuantizedElementType(signed_tensor_type);
  if (!qtype || !qtype.isSigned()) return {};

  int num_bits = qtype.getStorageTypeIntegralWidth();
  // This is a negative value, and will be applied on zero points and fixed
  // point ranges.
  int64_t offset =
      QType::getDefaultMinimumForInteger(/*isSigned=*/true, num_bits) -
      QType::getDefaultMinimumForInteger(/*isSigned=*/false, num_bits);

  auto flags = !quant::QuantizationFlags::Signed;
  QType new_qtype;
  if (auto uqtype = qtype.dyn_cast<quant::UniformQuantizedType>()) {
    new_qtype = quant::UniformQuantizedType::getChecked(
        loc, flags, qtype.getStorageType(), qtype.getExpressedType(),
        uqtype.getScale(), uqtype.getZeroPoint() - offset,
        uqtype.getStorageTypeMin() - offset,
        uqtype.getStorageTypeMax() - offset);
  } else if (auto aqtype =
                 qtype.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
    auto zero_points = aqtype.getZeroPoints();
    llvm::SmallVector<int64_t, 4> new_zero_points(zero_points.begin(),
                                                  zero_points.end());
    for (int i = 0, e = new_zero_points.size(); i != e; ++i) {
      new_zero_points[i] -= offset;
    }
    new_qtype = quant::UniformQuantizedPerAxisType::getChecked(
        loc, flags, qtype.getStorageType(), qtype.getExpressedType(),
        aqtype.getScales(), new_zero_points, aqtype.getQuantizedDimension(),
        aqtype.getStorageTypeMin() - offset,
        aqtype.getStorageTypeMax() - offset);
  }
  return new_qtype.castFromExpressedType(
      QType::castToExpressedType(signed_tensor_type));
}

LogicalResult RemoveDebugAttrPattern::matchAndRewrite(
    Operation* op, PatternRewriter& rewriter) const {
  // removeAttr will return nullptr if the attribute did not exist. Thus we can
  // return success(result) to indicate if this op has changed.
  return success(/*isSuccess=*/
                 op->removeAttr(kDebugModeOpQuantAttrName) ||
                 op->removeAttr(kDebugModeOpFloatAttrName));
}

}  // namespace quant
}  // namespace mlir
