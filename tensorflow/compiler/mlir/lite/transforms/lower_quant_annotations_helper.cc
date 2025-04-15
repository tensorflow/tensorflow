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
#include <limits>
#include <string>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::TFL {

LogicalResult FillCompositeParams(stablehlo::CompositeOp op,
                                  SmallVector<double, 4>& scales,
                                  SmallVector<int64_t, 4>& zero_points,
                                  int& num_bits, bool& is_signed,
                                  bool& is_narrow_range) {
  auto scale_attr = llvm::dyn_cast_or_null<DenseFPElementsAttr>(
      op.getCompositeAttributes().get("scale"));
  if (scale_attr == nullptr) {
    return failure();
  }
  for (auto float_attr : scale_attr.getValues<FloatAttr>()) {
    scales.push_back(float_attr.getValue().convertToDouble());
  }

  auto zero_point_attr = llvm::dyn_cast_or_null<DenseIntElementsAttr>(
      op.getCompositeAttributes().get("zero_point"));
  if (zero_point_attr == nullptr) {
    for (int i = 0; i < scales.size(); ++i) {
      zero_points.push_back(0);
    }
  } else if (zero_point_attr.isSplat()) {
    for (int i = 0; i < scales.size(); ++i) {
      zero_points.push_back(
          zero_point_attr.getSplatValue<IntegerAttr>().getInt());
    }
  } else {
    for (IntegerAttr zp : zero_point_attr.getValues<IntegerAttr>()) {
      zero_points.push_back(zp.getInt());
    }
  }

  auto dtype_attr = llvm::dyn_cast_or_null<StringAttr>(
      op.getCompositeAttributes().get("dtype"));
  if (dtype_attr == nullptr) {
    return failure();
  }
  std::string dtype = dtype_attr.getValue().str();
  if (dtype == "i8") {
    num_bits = 8;
    is_signed = true;
  } else if (dtype == "i4") {
    num_bits = 4;
    is_signed = true;
  } else {
    return failure();
  }
  auto narrow_range_attr = llvm::dyn_cast_or_null<BoolAttr>(
      op.getCompositeAttributes().get("narrow_range"));
  if (narrow_range_attr == nullptr) {
    return failure();
  }
  is_narrow_range = narrow_range_attr.getValue();

  return success();
}

bool IsDrqFakeQuant(stablehlo::CompositeOp op) {
  if (op.getName() != "quant.fake_quant") {
    return false;
  }
  SmallVector<double, 4> scales;
  SmallVector<int64_t, 4> zero_points;
  int num_bits;
  bool is_signed;
  bool is_narrow_range;
  if (failed(FillCompositeParams(op, scales, zero_points, num_bits, is_signed,
                                 is_narrow_range))) {
    return false;
  }
  return scales.empty() && zero_points.empty();
}

LogicalResult GetStorageParams(unsigned num_bits, bool narrow_range,
                               bool is_signed, MLIRContext* ctx,
                               Type& storage_type, int64_t& qmin,
                               int64_t& qmax) {
  if (num_bits <= 4) {
    storage_type = IntegerType::get(ctx, 4);
    if (is_signed) {
      qmin = -8;
      qmax = 7;
    } else {
      qmin = 0;
      qmax = 15;
    }
  } else if (num_bits <= 8) {
    storage_type = IntegerType::get(ctx, 8);
    if (is_signed) {
      qmin = -128;
      qmax = 127;
    } else {
      qmin = 0;
      qmax = 255;
    }
  } else if (num_bits <= 16) {
    storage_type = IntegerType::get(ctx, 16);
    if (is_signed) {
      qmin = -32768;
      qmax = 32767;
    } else {
      qmin = 0;
      qmax = 65535;
    }
  } else if (num_bits <= 32) {
    storage_type = IntegerType::get(ctx, 32);
    if (is_signed) {
      qmin = std::numeric_limits<int32_t>::min();
      qmax = std::numeric_limits<int32_t>::max();
    } else {
      qmin = std::numeric_limits<uint32_t>::min();
      qmax = std::numeric_limits<uint32_t>::max();
    }
  } else {
    return failure();
  }

  // Handle narrow_range.
  if (narrow_range) {
    qmin += 1;
  }
  return success();
}

Type GetPerTensorQuantizedTensorType(Builder& builder, double scale,
                                     int64_t zero_point, Type expressed_type,
                                     int num_bits, Location loc,
                                     bool narrow_range, bool is_signed) {
  unsigned flags = is_signed ? quant::QuantizationFlags::Signed : 0;
  MLIRContext* ctx = builder.getContext();
  Type storage_type;
  int64_t qmin;
  int64_t qmax;
  if (failed(GetStorageParams(num_bits, narrow_range, is_signed, ctx,
                              storage_type, qmin, qmax))) {
    return (emitError(loc, "unsupported FakeQuant number of bits: ")
                << num_bits,
            nullptr);
  }

  return quant::UniformQuantizedType::getChecked(
      loc, flags, storage_type, expressed_type, scale, zero_point, qmin, qmax);
}

Type GetPerAxisQuantizedTensorType(Builder& builder,
                                   SmallVector<double, 4> scales,
                                   SmallVector<int64_t, 4> zero_points,
                                   int32_t quantized_dimension,
                                   Type expressed_type, int num_bits,
                                   Location loc, bool narrow_range,
                                   bool is_signed) {
  unsigned flags = is_signed ? quant::QuantizationFlags::Signed : 0;

  MLIRContext* ctx = builder.getContext();
  Type storage_type;
  int64_t qmin;
  int64_t qmax;
  if (failed(GetStorageParams(num_bits, narrow_range, is_signed, ctx,
                              storage_type, qmin, qmax))) {
    return (emitError(loc, "unsupported FakeQuant number of bits: ")
                << num_bits,
            nullptr);
  }

  return quant::UniformQuantizedPerAxisType::getChecked(
      loc, flags, storage_type, expressed_type, scales, zero_points,
      quantized_dimension, qmin, qmax);
}

}  // namespace mlir::TFL
