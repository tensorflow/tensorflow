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

#include <cstdint>

#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantizeUtils.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/UniformSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"

namespace mlir {
namespace TFL {

// Returns the quantized type for the
// input_type/min/max/storag_type_width/narrow_range.
static Type GetQuantizedType(Builder builder, Type input_type,
                             ArrayRef<double> min, ArrayRef<double> max,
                             int storage_type_width, bool narrow_range,
                             bool is_signed) {
  auto converter =
      quant::ExpressedToQuantizedConverter::forInputType(input_type);

  quant::QuantizedType quantizedEleType;
  if (min.size() == 1 && max.size() == 1) {
    quantizedEleType = quant::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, min[0], max[0],
        narrow_range, converter.expressedType, is_signed);
  } else if (min.size() == max.size()) {
    auto shape = input_type.dyn_cast<ShapedType>();
    if (!shape || min.size() != shape.getDimSize(shape.getRank() - 1)) {
      return {};
    }
    quantizedEleType = quant::fakeQuantAttrsToType(
        builder.getUnknownLoc(), storage_type_width, shape.getRank() - 1, min,
        max, narrow_range, converter.expressedType, is_signed);
  }
  if (!quantizedEleType) return {};
  return converter.convert(quantizedEleType);
}

TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, FloatAttr min,
                              FloatAttr max, Type storage_type,
                              bool narrow_range, bool is_signed) {
  int storage_type_width = storage_type.cast<IntegerType>().getWidth();
  Type final_type = GetQuantizedType(
      builder, input_type, {min.getValueAsDouble()}, {max.getValueAsDouble()},
      storage_type_width, narrow_range, is_signed);
  return builder.getTypeAttr(final_type);
}

TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, IntegerAttr num_bits,
                              BoolAttr narrow_range, bool is_signed) {
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
      GetQuantizedType(builder, input_type, min_value, max_value,
                       num_bits.getInt(), narrow_range.getValue(), is_signed);
  return builder.getTypeAttr(final_type);
}

TypeAttr CastQuantizedTypeAttrFromExpressedType(Builder builder,
                                                TypeAttr source, Type target) {
  if (!source || !source.getValue().isa<TensorType>()) return {};
  auto ele_type = source.getValue().cast<TensorType>().getElementType();
  if (auto quantized_type = ele_type.dyn_cast<quant::QuantizedType>()) {
    Type final_type = quantized_type.castFromExpressedType(target);
    if (final_type) return builder.getTypeAttr(final_type);
  }
  return {};
}

Type GetUniformQuantizedTypeForElementsAttr(ElementsAttr attr,
                                            unsigned storage_type_width,
                                            bool is_signed, bool narrow_range) {
  Builder builder(attr.getContext());
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();
  auto fp = attr.dyn_cast<DenseFPElementsAttr>();
  if (!fp) return {};

  // If all the element values are same we don't need to scan the content.
  if (fp.isSplat()) {
    double single_value =
        FloatAttr::getValueAsDouble(fp.getSplatValue<llvm::APFloat>());
    // the mlir quantization libration can only handle the case min=max=0.0, so
    // we just avoid quantization if it is any values other than 0.0.
    // TODO(b/141015060): remove this constraint once the bug is fixed.
    if (std::fabs(single_value) > std::numeric_limits<double>::epsilon()) {
      return {};
    }
    min = max = single_value;
  } else {
    for (auto it = fp.begin(), e = fp.end(); it != e; ++it) {
      double ele_value = FloatAttr::getValueAsDouble(*it);
      min = std::min(min, ele_value);
      max = std::max(max, ele_value);
    }
  }
  auto type = GetQuantizedType(builder, attr.getType(), min, max,
                               storage_type_width, narrow_range, is_signed);
  if (auto ele_type = type.dyn_cast_or_null<TensorType>())
    return ele_type.getElementType();

  return {};
}

quant::QuantizedType GetUniformQuantizedTypeForBias(
    const std::vector<quant::QuantizedType>& op_types) {
  if (op_types.empty()) return {};

  double scale = 1.0;
  for (unsigned i = 0, e = op_types.size(); i != e; ++i) {
    auto qtype = op_types[i].dyn_cast_or_null<quant::UniformQuantizedType>();
    if (!qtype) return {};
    scale *= qtype.getScale();
  }
  auto type = op_types.back().cast<quant::UniformQuantizedType>();
  Builder builder(type.getContext());
  // TODO(fengliuai): make the bit width configurable.
  IntegerType storageType = builder.getIntegerType(32);
  return quant::UniformQuantizedType::getChecked(
      /*flags=*/true, storageType, type.getExpressedType(), scale,
      /*zeroPoint=*/0,
      quant::QuantizedType::getDefaultMininumForInteger(/*isSigned=*/true, 32),
      quant::QuantizedType::getDefaultMaxinumForInteger(/*isSigned=*/true, 32),
      builder.getUnknownLoc());
}

ElementsAttr Quantize(Attribute real_value, Type tensor_type) {
  if (auto q_type =
          quant::QuantizedType::getQuantizedElementType(tensor_type)) {
    Type converted_type;
    return quant::quantizeAttr(real_value, q_type, converted_type)
        .cast<ElementsAttr>();
  }
  return {};
}

}  // namespace TFL
}  // namespace mlir
