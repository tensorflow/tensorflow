/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/lower_quant_annotations_helper.h"

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::TFL {

LogicalResult FillCompositeParams(stablehlo::CompositeOp op,
                                  SmallVector<double, 4>& scales,
                                  SmallVector<int64_t, 4>& zero_points,
                                  int32_t& quantized_dimension, int& num_bits,
                                  bool& is_signed, int64_t& storage_type_min,
                                  int64_t& storage_type_max) {
  auto scale_attr = llvm::dyn_cast_or_null<DenseFPElementsAttr>(
      op.getCompositeAttributes().get("scale"));
  if (scale_attr == nullptr) {
    return failure();
  }
  scales.reserve(scale_attr.getNumElements());
  for (auto float_attr : scale_attr.getValues<FloatAttr>()) {
    scales.push_back(float_attr.getValueAsDouble());
  }

  zero_points.resize(scales.size(), 0);
  auto zero_point_attr = llvm::dyn_cast_or_null<DenseIntElementsAttr>(
      op.getCompositeAttributes().get("zero_point"));
  if (zero_point_attr) {
    auto temp_vec = llvm::to_vector(zero_point_attr.getValues<int64_t>());
    zero_points.assign(temp_vec.begin(), temp_vec.end());
  }

  quantized_dimension = -1;
  auto quantDimensionAttr = llvm::dyn_cast_or_null<IntegerAttr>(
      op.getCompositeAttributes().get("quantization_dimension"));
  if (quantDimensionAttr) {
    quantized_dimension = quantDimensionAttr.getValue().getSExtValue();
  }

  if (op.getName() == "quant.quantize") {
    auto elementType = getElementTypeOrSelf(op.getResults().front().getType());
    num_bits = elementType.getIntOrFloatBitWidth();
    is_signed = !elementType.isUnsignedInteger();
  } else if (op.getName() == "quant.dequantize") {
    auto elementType = getElementTypeOrSelf(op.getInputs().front().getType());
    if (auto quantized_element_type =
            dyn_cast<quant::QuantizedType>(elementType)) {
      num_bits = quantized_element_type.getStorageTypeIntegralWidth();
      is_signed = quantized_element_type.isSigned();
    } else {
      num_bits = elementType.getIntOrFloatBitWidth();
      is_signed = !elementType.isUnsignedInteger();
    }
  } else if (op.getName() == "quant.fake_quant") {
    auto dtype_attr = llvm::dyn_cast_or_null<TypeAttr>(
        op.getCompositeAttributes().get("dtype"));
    if (!dtype_attr) {
      return failure();
    }
    auto dtype = dtype_attr.getValue();
    if (dtype.isInteger(8)) {
      num_bits = 8;
      is_signed = true;
    } else if (dtype.isInteger(4)) {
      num_bits = 4;
      is_signed = true;
    } else {
      return failure();
    }
  } else {
    return failure();
  }

  auto storage_type_min_attr = llvm::dyn_cast_or_null<IntegerAttr>(
      op.getCompositeAttributes().get("storage_type_min"));
  if (storage_type_min_attr == nullptr) {
    return failure();
  }
  storage_type_min = storage_type_min_attr.getValue().getSExtValue();

  auto storage_type_max_attr = llvm::dyn_cast_or_null<IntegerAttr>(
      op.getCompositeAttributes().get("storage_type_max"));
  if (storage_type_max_attr == nullptr) {
    return failure();
  }
  storage_type_max = storage_type_max_attr.getValue().getSExtValue();

  return success();
}

Type GetPerTensorQuantizedTensorType(Builder& builder, double scale,
                                     int64_t zero_point, Type expressed_type,
                                     int num_bits, Location loc,
                                     int64_t storage_type_min,
                                     int64_t storage_type_max, bool is_signed) {
  unsigned flags = is_signed ? quant::QuantizationFlags::Signed : 0;
  MLIRContext* ctx = builder.getContext();
  Type storage_type = IntegerType::get(ctx, num_bits);
  return quant::UniformQuantizedType::getChecked(
      loc, flags, storage_type, expressed_type, scale, zero_point,
      storage_type_min, storage_type_max);
}

Type GetPerAxisQuantizedTensorType(Builder& builder,
                                   SmallVector<double, 4> scales,
                                   SmallVector<int64_t, 4> zero_points,
                                   int32_t quantized_dimension,
                                   Type expressed_type, int num_bits,
                                   Location loc, int64_t storage_type_min,
                                   int64_t storage_type_max, bool is_signed) {
  unsigned flags = is_signed ? quant::QuantizationFlags::Signed : 0;

  MLIRContext* ctx = builder.getContext();
  Type storage_type = IntegerType::get(ctx, num_bits);

  return quant::UniformQuantizedPerAxisType::getChecked(
      loc, flags, storage_type, expressed_type, scales, zero_points,
      quantized_dimension, storage_type_min, storage_type_max);
}

}  // namespace mlir::TFL
