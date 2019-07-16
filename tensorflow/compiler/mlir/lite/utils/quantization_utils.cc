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

#include "tensorflow/compiler/mlir/lite/utils/quantization_utils.h"

#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/QuantizeUtils.h"  // TF:local_config_mlir
#include "mlir/Dialect/QuantOps/UniformSupport.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"

namespace mlir {
namespace TFL {

// Returns the quantized type for the
// input_type/min/max/storag_type_width/narrow_range.
static Type GetQuantizedType(Builder builder, Type input_type, double min,
                             double max, int storage_type_width,
                             bool narrow_range, bool is_signed) {
  auto converter =
      quant::ExpressedToUniformQuantizedConverter::forInputType(input_type);

  quant::UniformQuantizedType quantizedEleType = quant::fakeQuantAttrsToType(
      builder.getUnknownLoc(), storage_type_width, min, max, narrow_range,
      converter.expressedType, is_signed);
  return converter.convert(quantizedEleType);
}

TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, FloatAttr min,
                              FloatAttr max, Type storage_type,
                              bool narrow_range, bool is_signed) {
  int storage_type_width = storage_type.cast<IntegerType>().getWidth();
  Type final_type = GetQuantizedType(
      builder, input_type, min.getValueAsDouble(), max.getValueAsDouble(),
      storage_type_width, narrow_range, is_signed);
  return builder.getTypeAttr(final_type);
}

TypeAttr GetQuantizedTypeAttr(Builder builder, Type input_type, Attribute min,
                              Attribute max, IntegerAttr num_bits,
                              BoolAttr narrow_range) {
  FloatAttr min_value = GetSingleElementAsFloatOrSelf(min);
  FloatAttr max_value = GetSingleElementAsFloatOrSelf(max);
  if (!min_value || !max_value) return {};
  return GetQuantizedTypeAttr(builder, input_type, min_value, max_value,
                              builder.getIntegerType(num_bits.getInt()),
                              narrow_range.getValue(), /*is_signed=*/false);
}

Type GetUniformQuantizedTypeForElementsAttr(ElementsAttr attr,
                                            unsigned storage_type_width,
                                            bool is_signed, bool narrow_range) {
  Builder builder(attr.getContext());
  double min = std::numeric_limits<double>::max();
  double max = std::numeric_limits<double>::min();
  if (auto fp = attr.dyn_cast<DenseFPElementsAttr>()) {
    for (auto it = fp.begin(), e = fp.end(); it != e; ++it) {
      double ele_value = FloatAttr::getValueAsDouble(*it);
      min = std::min(min, ele_value);
      max = std::max(max, ele_value);
    }
    // The range must straddle zero.
    if (min > 0.0 || max < 0.0) return {};
    auto type = GetQuantizedType(builder, attr.getType(), min, max,
                                 storage_type_width, narrow_range, is_signed);
    if (auto ele_type = type.dyn_cast_or_null<TensorType>())
      return ele_type.getElementType();
  }

  // The range from SplatElementAttr and other element attribute types  couldn't
  // straddle zero, so the quantization parameters couldn't be derived from its
  // range.
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
