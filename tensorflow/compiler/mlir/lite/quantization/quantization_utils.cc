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
#include <limits>
#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Quant/FakeQuantSupport.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantizeUtils.h"  // from @llvm-project
#include "mlir/Dialect/Quant/UniformSupport.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace quant {

const float kNearZeroTolerance = 1.0e-6;

// Returns the quantized type for the
// input_type/min/max/storag_type_width/narrow_range.
static Type GetQuantizedType(Builder builder, Type input_type,
                             ArrayRef<double> min, ArrayRef<double> max,
                             int quant_dim, int storage_type_width,
                             bool narrow_range, bool is_signed) {
  auto converter =
      quant::ExpressedToQuantizedConverter::forInputType(input_type);

  quant::QuantizedType quantizedEleType;
  if (min.size() == 1 && max.size() == 1 && quant_dim == -1) {
    quantizedEleType = quant::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, min[0], max[0],
        narrow_range, converter.expressedType, is_signed);
  } else if (min.size() == max.size()) {
    auto shape = input_type.dyn_cast<ShapedType>();
    if (!shape || shape.getRank() <= quant_dim ||
        static_cast<int64_t>(min.size()) != shape.getDimSize(quant_dim)) {
      return {};
    }
    // TODO(b/141508873): the quantization dim is set to the last dimension.
    quantizedEleType = quant::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, quant_dim, min, max,
        narrow_range, converter.expressedType, is_signed);
  }
  if (!quantizedEleType) return {};
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
    if (static_cast<int64_t>(scales.size()) != factor_values.getNumElements()) return {};
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
                              bool is_signed) {
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
                       num_bits.getInt(), narrow_range.getValue(), is_signed);
  if (!final_type) return {};
  return TypeAttr::get(final_type);
}

// Repeats the content of `data` multiple times to resize to `target_size`.
// Note that this only broadcast across one dimension.
template <typename T>
static bool BroadcastVector(int target_size, SmallVectorImpl<T>& data) {
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
static quant::UniformQuantizedPerAxisType ResetAxisAndBroadcast(
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
  } else if ((new_shape.size() == shape.size() + 1) && new_shape.back() == 1) {
    // This is a trivial shift left, then we shift the quant_dim as well.
    if (std::equal(shape.begin(), shape.end(), new_shape.begin()) &&
        quant_dim == -1) {
      quant_dim = shape.size() + quant_dim;
    } else {
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
    qtype =
        ResetAxisAndBroadcast(source_type.getShape(), per_axis, target, axis);
  }
  if (!qtype) return {};
  Type final_type = qtype.castFromExpressedType(target);
  if (!final_type) return {};
  return TypeAttr::get(final_type);
}

Type GetUniformQuantizedTypeForWeight(ElementsAttr attr, bool symmetric,
                                      unsigned num_bits, bool is_signed,
                                      bool narrow_range) {
  Builder builder(attr.getContext());
  // `symmetric` can only be used when it is `signed` and `narrow_range`.
  if (symmetric && (!is_signed || !narrow_range)) return {};

  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();
  auto fp = attr.dyn_cast<DenseFPElementsAttr>();
  if (!fp) return {};

  // If all the element values are same we don't need to scan the content.
  if (fp.isSplat()) {
    double single_value =
        FloatAttr::getValueAsDouble(fp.getSplatValue<llvm::APFloat>());
    // When the single value isn't 0.0, we expand it to a range to include
    // this single value and 0.0. This will give us a scale and zero point
    // works for both this value and 0.0.
    if (single_value < 0.0) {
      min = single_value;
      max = symmetric ? -single_value : 0.0;
    } else if (single_value > 0.0) {
      min = symmetric ? -single_value : 0.0;
      max = single_value;
    } else {
      min = max = single_value;
    }
  } else {
    for (auto it = fp.begin(), e = fp.end(); it != e; ++it) {
      double ele_value = FloatAttr::getValueAsDouble(*it);
      min = std::min(min, ele_value);
      max = std::max(max, ele_value);
      if (symmetric) {
        max = std::max(std::abs(min), std::abs(max));
        // In case the scale is extremely small, a fixed scale is used.
        if (max < kNearZeroTolerance) max = 1.0;
        min = -max;
      }
    }
  }
  auto type =
      GetQuantizedType(builder, attr.getType(), min, max, /*quant_dim=*/-1,
                       num_bits, narrow_range, is_signed);
  if (auto ele_type = type.dyn_cast_or_null<TensorType>())
    return ele_type.getElementType();

  return {};
}

Type GetUniformQuantizedPerAxisTypeForWeight(ElementsAttr attr, int quant_dim,
                                             bool symmetric, unsigned num_bits,
                                             bool is_signed,
                                             bool narrow_range) {
  Builder builder(attr.getContext());
  auto shape = attr.getType().cast<ShapedType>().getShape();
  if (static_cast<int>(shape.size()) <= quant_dim) return {};
  // `symmetric` can only be used when it is `signed` and `narrow_range`.
  if (symmetric && (!is_signed || !narrow_range)) return {};

  int dim_size = shape[quant_dim];
  int slice_size = std::accumulate(std::next(shape.begin(), quant_dim + 1),
                                   shape.end(), 1, std::multiplies<int64_t>());
  SmallVector<double, 4> min(dim_size, std::numeric_limits<double>::max());
  SmallVector<double, 4> max(dim_size, std::numeric_limits<double>::min());
  auto fp = attr.dyn_cast<DenseFPElementsAttr>();
  if (!fp) return {};

  // If all the element values are same we don't need to scan the content.
  if (fp.isSplat()) {
    double single_value =
        FloatAttr::getValueAsDouble(fp.getSplatValue<llvm::APFloat>());
    // When the single value isn't 0.0, we expand it to a range to include
    // this single value and 0.0. This will give us a scale and zero point
    // works for both this value and 0.0.
    if (single_value < 0.0) {
      min[0] = single_value;
      max[0] = symmetric ? -single_value : 0.0;
    } else if (single_value > 0.0) {
      min[0] = symmetric ? -single_value : 0.0;
      max[0] = single_value;
    } else {
      // If the tensor contents are all 0.0f, a fixed range is used to avoid
      // INF error.
      min[0] = -1.0;
      max[0] = 1.0;
    }
    for (int i = 1; i < dim_size; ++i) {
      min[i] = min[0];
      max[i] = max[0];
    }
  } else {
    int64_t flatten_index = 0;
    for (auto it = fp.begin(), e = fp.end(); it != e; ++it, ++flatten_index) {
      double ele_value = FloatAttr::getValueAsDouble(*it);
      int slice_index = flatten_index / slice_size;
      int channel_index = slice_index % dim_size;
      min[channel_index] = std::min(min[channel_index], ele_value);
      max[channel_index] = std::max(max[channel_index], ele_value);
    }
    if (symmetric) {
      for (int i = 0; i < dim_size; ++i) {
        max[i] = std::max(std::abs(min[i]), std::abs(max[i]));
        // In case the scale is extremely small, a fixed scale is used.
        if (max[i] < kNearZeroTolerance) max[i] = 1.0;
        min[i] = -max[i];
      }
    }
  }
  auto type = GetQuantizedType(builder, attr.getType(), min, max, quant_dim,
                               num_bits, narrow_range, is_signed);
  if (auto ele_type = type.dyn_cast_or_null<TensorType>())
    return ele_type.getElementType();

  return {};
}

quant::QuantizedType GetUniformQuantizedTypeForBias(
    const std::vector<quant::QuantizedType>& op_types) {
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
      for (auto index_scale : llvm::enumerate(type.getScales())) {
        scales[index_scale.index()] *= index_scale.value();
      }
    } else if (auto type = op_type.dyn_cast<quant::UniformQuantizedType>()) {
      for (size_t index = 0; index != axis_size; ++index) {
        scales[index] *= type.getScale();
      }
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
        /*flags=*/true, storage_type, expressed_type, scales[0],
        /*zeroPoint=*/0, storage_type_min, storage_type_max,
        builder.getUnknownLoc());
  } else {
    llvm::SmallVector<int64_t, 4> zero_points(axis_size, 0);
    // TODO(b/141508873): Assume the bias is a 1-D tensor, and set the
    // quantization dim to the last dimension, which is 0. If the bias rank is
    // larger than 1, this returned quantized type couldn't be used to
    // quantize the bias.
    return quant::UniformQuantizedPerAxisType::getChecked(
        /*flags=*/true, storage_type, expressed_type, scales, zero_points,
        /*quantizedDimension=*/0, storage_type_min, storage_type_max,
        builder.getUnknownLoc());
  }
}

ElementsAttr Quantize(Attribute real_value, Type tensor_type) {
  if (auto q_type =
          quant::QuantizedType::getQuantizedElementType(tensor_type)) {
    Type converted_type;
    return quant::quantizeAttr(real_value, q_type, converted_type)
        .dyn_cast<ElementsAttr>();
  }
  return {};
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

// The stats op of some of the ops can be redundant. The current implementation
// only considers the ops with restricted output params.
static bool IsStatsRedundant(Operation* op,
                             OpQuantSpecGetter op_quant_spec_getter) {
  return !op_quant_spec_getter(op)->restricted_output_params.empty();
}

bool RemoveRedundantStatsOps(mlir::FuncOp func,
                             OpQuantSpecGetter op_quant_spec_getter) {
  llvm::SmallVector<quant::StatisticsOp, 16> all_stats_ops;
  llvm::DenseSet<Operation*> redundant_stats_ops;

  // Step 1: forward pass: propagate any value scales which are not produces
  // by `SameOperandsAndResultsScale`. Additionally, remove the value scales
  // which are produced by the `restricted_output_params`.
  // Note that we don't propagate across the multiple-operands
  // `SameOperandsAndResultsScale` ops like `concatenation`.
  func.walk(
      [&](quant::StatisticsOp stats_op) { all_stats_ops.push_back(stats_op); });

  while (!all_stats_ops.empty()) {
    quant::StatisticsOp stats_op = all_stats_ops.back();
    all_stats_ops.pop_back();

    if (auto def = stats_op.arg().getDefiningOp()) {
      if (IsStatsRedundant(def, op_quant_spec_getter)) {
        redundant_stats_ops.insert(stats_op);
      }
    }

    for (auto user : stats_op.getResult().getUsers()) {
      // We don't propagate this parameter down if it has multiple operands.
      // We want to use the result parameter scales instead.

      if (user->hasTrait<OpTrait::quant::SameOperandsAndResultsScale>() &&
          !PreferResultScale(user)) {
        for (Value res : user->getResults()) {
          if (res.hasOneUse()) {
            if (auto next_stats = llvm::dyn_cast<quant::StatisticsOp>(
                    *res.getUsers().begin())) {
              // quantization parameters can be propagated to next_stats
              redundant_stats_ops.insert(next_stats);
              // add next_stats to the work list so propagation can
              // continue.
              all_stats_ops.push_back(next_stats);
            }
          }
        }
      }
    }
  }

  // Step 2: backward pass: For the ops skiped in the forward pass, propagate
  // its results scale backwards as far as possible.
  func.walk([&](quant::StatisticsOp stats_op) {
    if (redundant_stats_ops.find(stats_op) == redundant_stats_ops.end()) {
      all_stats_ops.push_back(stats_op);
    }
  });

  while (!all_stats_ops.empty()) {
    quant::StatisticsOp stats_op = all_stats_ops.back();
    all_stats_ops.pop_back();

    if (auto def = stats_op.arg().getDefiningOp()) {
      if (def->hasTrait<OpTrait::quant::SameOperandsAndResultsScale>()) {
        for (auto input : def->getOperands()) {
          if (auto next_stats = llvm::dyn_cast_or_null<quant::StatisticsOp>(
                  input.getDefiningOp())) {
            redundant_stats_ops.insert(next_stats);
            all_stats_ops.push_back(next_stats);
          }
        }
      }
    }
  }

  // Step3: Remove all the redundant stats ops
  for (auto it : redundant_stats_ops) {
    if (!llvm::isa<quant::StatisticsOp>(it)) return true;
    auto stats_op = llvm::cast<quant::StatisticsOp>(it);
    stats_op.getResult().replaceAllUsesWith(stats_op.arg());
    stats_op.erase();
  }

  // Returns false if the steps finish without errors.
  return false;
}
}  // namespace quant
}  // namespace mlir
