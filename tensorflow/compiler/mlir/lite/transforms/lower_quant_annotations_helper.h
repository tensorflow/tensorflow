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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_QUANT_ANNOTATIONS_HELPER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_QUANT_ANNOTATIONS_HELPER_H_

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::TFL {

// Quantization annotations reach this pass as either a `stablehlo.composite` or
// a `stablehlo.custom_call`. The two carry the same information but spell the
// accessors differently, so these overloads let the templates below handle both
// without duplicating the extraction logic.
inline Attribute GetOpAttribute(stablehlo::CompositeOp op, StringRef name) {
  return op.getCompositeAttributes().get(name);
}

inline Attribute GetOpAttribute(stablehlo::CustomCallOp op, StringRef name) {
  return op->getAttr(name);
}

inline StringRef GetQuantAnnotationName(stablehlo::CompositeOp op) {
  return op.getName();
}

inline StringRef GetQuantAnnotationName(stablehlo::CustomCallOp op) {
  return op.getCallTargetName();
}

// The integer storage of a quantized tensor, as named by the annotation's
// `dtype` attribute.
struct QuantStorage {
  Type type;
  int num_bits = 0;
  bool is_signed = false;

  // Inclusive bounds of the representable range. `narrow_range` drops the most
  // negative value so that the range is symmetric around zero.
  //
  // This is the single definition of the storage range used by the whole
  // lowering: the constant folder that quantizes weights at compile time and
  // the runtime kernels must agree on it exactly, or the model's numerics
  // change depending on which one ran.
  int64_t Min(bool narrow_range) const {
    if (!is_signed) return 0;
    const int64_t min = -(int64_t{1} << (num_bits - 1));
    return narrow_range ? min + 1 : min;
  }
  int64_t Max(bool narrow_range) const {
    if (!is_signed) return (int64_t{1} << num_bits) - 1;
    return (int64_t{1} << (num_bits - 1)) - 1;
  }
};

// Parses the annotation's `dtype` string. Fails on an unrecognized name rather
// than defaulting, because every caller uses the result to decide how to round
// and clamp real values.
inline LogicalResult ParseQuantStorage(StringRef dtype, MLIRContext* ctx,
                                       QuantStorage& storage) {
  if (dtype == "i2") {
    storage = {IntegerType::get(ctx, 2), 2, /*is_signed=*/true};
  } else if (dtype == "i4") {
    storage = {IntegerType::get(ctx, 4), 4, /*is_signed=*/true};
  } else if (dtype == "ui4") {
    storage = {IntegerType::get(ctx, 4), 4, /*is_signed=*/false};
  } else if (dtype == "i8") {
    storage = {IntegerType::get(ctx, 8), 8, /*is_signed=*/true};
  } else if (dtype == "ui8") {
    storage = {IntegerType::get(ctx, 8), 8, /*is_signed=*/false};
  } else if (dtype == "i16") {
    storage = {IntegerType::get(ctx, 16), 16, /*is_signed=*/true};
  } else {
    return failure();
  }
  return success();
}

// Everything a `quant.*` annotation says about a blockwise quantized tensor.
//
// All quantization parameters are annotation attributes rather than operands:
// the annotation op always has exactly one operand, the value being quantized.
struct BlockwiseAnnotation {
  SmallVector<int64_t> block_shape;
  QuantStorage storage;
  bool narrow_range = false;
  bool symmetric = false;
  // Element type of the scale. `f8E8M0FNU` constrains the scale to a power of
  // two; otherwise `f32`.
  Type scale_type;
  // Widens the quantized range used to derive a dynamic scale, so that
  // outliers clip rather than stretching the whole block. 0 when absent.
  double range_dilation = 0.0;
  // Null for dynamic range quantization, where the scale is derived from the
  // tensor at runtime.
  ElementsAttr scale;
  // Null when the quantization is symmetric.
  ElementsAttr zero_point;

  bool IsDynamic() const { return scale == nullptr; }
};

// Returns true if `op` carries a `block_shape` attribute, i.e. it describes
// blockwise rather than per-tensor or per-axis quantization.
template <typename OpType>
bool IsBlockwiseAnnotation(OpType op) {
  return llvm::isa_and_nonnull<DenseIntElementsAttr>(
      GetOpAttribute(op, "block_shape"));
}

// Extracts the blockwise quantization parameters of `op`. Only call this when
// `IsBlockwiseAnnotation(op)` holds.
//
// Fails on any attribute that is present but unrecognized. Falling back to a
// default would produce a model that runs but computes something other than
// what the annotation asked for.
template <typename OpType>
LogicalResult ParseBlockwiseAnnotation(OpType op, BlockwiseAnnotation& out) {
  MLIRContext* ctx = op->getContext();

  auto block_shape_attr = llvm::dyn_cast_or_null<DenseIntElementsAttr>(
      GetOpAttribute(op, "block_shape"));
  if (block_shape_attr == nullptr) return failure();
  for (const IntegerAttr block :
       block_shape_attr.template getValues<IntegerAttr>()) {
    out.block_shape.push_back(block.getInt());
  }

  auto dtype_attr =
      llvm::dyn_cast_or_null<StringAttr>(GetOpAttribute(op, "dtype"));
  if (dtype_attr == nullptr) return failure();
  if (failed(ParseQuantStorage(dtype_attr.getValue(), ctx, out.storage))) {
    return failure();
  }

  if (auto narrow_range = llvm::dyn_cast_or_null<BoolAttr>(
          GetOpAttribute(op, "narrow_range"))) {
    out.narrow_range = narrow_range.getValue();
  }
  if (auto symmetric =
          llvm::dyn_cast_or_null<BoolAttr>(GetOpAttribute(op, "symmetric"))) {
    out.symmetric = symmetric.getValue();
  }

  out.scale_type = Float32Type::get(ctx);
  if (Attribute scale_dtype = GetOpAttribute(op, "act_scale_dtype")) {
    auto scale_dtype_str = llvm::dyn_cast<StringAttr>(scale_dtype);
    if (scale_dtype_str == nullptr) return failure();
    if (scale_dtype_str.getValue() == "e8m0") {
      out.scale_type = Float8E8M0FNUType::get(ctx);
    } else if (scale_dtype_str.getValue() != "fp32" &&
               scale_dtype_str.getValue() != "float32") {
      return failure();
    }
  }

  if (Attribute dilation = GetOpAttribute(op, "range_dilation")) {
    auto dilation_fp = llvm::dyn_cast<FloatAttr>(dilation);
    if (dilation_fp == nullptr) return failure();
    out.range_dilation = dilation_fp.getValueAsDouble();
  }

  // An empty `scale` means dynamic range quantization; see `IsDrqFakeQuant`.
  auto scale =
      llvm::dyn_cast_or_null<ElementsAttr>(GetOpAttribute(op, "scale"));
  if (scale != nullptr && !scale.empty()) out.scale = scale;

  auto zero_point =
      llvm::dyn_cast_or_null<ElementsAttr>(GetOpAttribute(op, "zero_point"));
  if (zero_point != nullptr && !zero_point.empty()) out.zero_point = zero_point;

  return success();
}

// Returns true if `op` carries the static `scale` attribute that describes its
// quantization parameters, i.e. the parameters are baked into the annotation.
//
// An annotation without one describes dynamic range quantization (DRQ), where
// the parameters are derived from the tensor at runtime. This is the exact
// condition under which `FillCompositeParams` below fails, so the two must stay
// in sync.
template <typename OpType>
bool HasStaticQuantParams(OpType op) {
  auto scale_attr =
      llvm::dyn_cast_or_null<DenseFPElementsAttr>(GetOpAttribute(op, "scale"));
  return scale_attr != nullptr && !scale_attr.empty();
}

// Extracts the quantization parameters from a per-tensor or per-axis `quant.*`
// annotation into `scales`, `zero_points`, `num_bits`, `is_signed` and
// `is_narrow_range`.
//
// Fails if any required attribute is missing or unrecognized. In particular it
// fails on DRQ annotations, which carry no static `scale`; callers that need to
// handle those must check `IsDrqFakeQuant`/`HasStaticQuantParams` first. On
// success `scales` is guaranteed non-empty and `zero_points` has the same size.
template <typename OpType>
LogicalResult FillCompositeParams(OpType op, SmallVector<double, 4>& scales,
                                  SmallVector<int64_t, 4>& zero_points,
                                  int& num_bits, bool& is_signed,
                                  bool& is_narrow_range) {
  auto scale_attr =
      llvm::dyn_cast_or_null<DenseFPElementsAttr>(GetOpAttribute(op, "scale"));
  if (scale_attr == nullptr || scale_attr.empty()) {
    return failure();
  }
  for (auto float_attr : scale_attr.template getValues<FloatAttr>()) {
    scales.push_back(float_attr.getValue().convertToDouble());
  }

  auto zero_point_attr = llvm::dyn_cast_or_null<DenseIntElementsAttr>(
      GetOpAttribute(op, "zero_point"));
  if (zero_point_attr == nullptr) {
    for (int i = 0; i < scales.size(); ++i) {
      zero_points.push_back(0);
    }
  } else if (zero_point_attr.isSplat()) {
    for (int i = 0; i < scales.size(); ++i) {
      zero_points.push_back(
          zero_point_attr.template getSplatValue<IntegerAttr>().getInt());
    }
  } else {
    for (IntegerAttr zp : zero_point_attr.template getValues<IntegerAttr>()) {
      zero_points.push_back(zp.getInt());
    }
  }

  auto dtype_attr =
      llvm::dyn_cast_or_null<StringAttr>(GetOpAttribute(op, "dtype"));
  if (dtype_attr == nullptr) {
    return failure();
  }
  QuantStorage storage;
  if (failed(ParseQuantStorage(dtype_attr.getValue(), op->getContext(),
                               storage))) {
    return failure();
  }
  num_bits = storage.num_bits;
  is_signed = storage.is_signed;

  auto narrow_range_attr =
      llvm::dyn_cast_or_null<BoolAttr>(GetOpAttribute(op, "narrow_range"));
  if (narrow_range_attr == nullptr) {
    return failure();
  }
  is_narrow_range = narrow_range_attr.getValue();

  return success();
}

template <typename OpType>
bool IsDrqFakeQuant(OpType op) {
  if (GetQuantAnnotationName(op) != "quant.fake_quant") {
    return false;
  }
  return !HasStaticQuantParams(op);
}

LogicalResult GetStorageParams(unsigned num_bits, bool narrow_range,
                               bool is_signed, MLIRContext* ctx,
                               Type& storage_type, int64_t& qmin,
                               int64_t& qmax);

Type GetPerTensorQuantizedTensorType(Builder& builder, double scale,
                                     int64_t zero_point, Type expressed_type,
                                     int num_bits, Location loc,
                                     bool narrow_range, bool is_signed);

Type GetPerAxisQuantizedTensorType(Builder& builder,
                                   SmallVector<double, 4> scales,
                                   SmallVector<int64_t, 4> zero_points,
                                   int32_t quantized_dimension,
                                   Type expressed_type, int num_bits,
                                   Location loc, bool narrow_range,
                                   bool is_signed);

}  // namespace mlir::TFL
#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_LOWER_QUANT_ANNOTATIONS_HELPER_H_
