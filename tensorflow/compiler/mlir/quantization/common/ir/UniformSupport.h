/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_UNIFORMSUPPORT_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_UNIFORMSUPPORT_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir::quantfork {

// Performs type conversion from an arbitrary input type to a type
// that is expressed by a QuantizedType.
//
// This handles cases where the inputType is a supported primitive type
// (i.e. f32, bf16, etc) or a vector/tensor type based on a supported
// elemental type.
//
// Since conversion often involves introspecting some attributes of the
// input type in order to determine how to represent it, this is a two step
// process.
struct ExpressedToQuantizedConverter {
  // Creates a converter for the given input type.
  static ExpressedToQuantizedConverter forInputType(Type input_type);

  // Converts the inputType to be based on the given elemental type,
  // returning the new type (or nullptr and emit an error on failure).
  Type convert(quant::QuantizedType elemental_type) const;

  // Whether the conversion is legal.
  explicit operator bool() const { return (bool)expressed_type; }

  // The input type that is being converted from.
  // This may be an elemental or composite type.
  const Type input_type;

  // Supported, elemental expressed type (i.e. f32).
  // Will be nullptr if conversion is not supported.
  const Type expressed_type;
};

// Reference implementation of converting between real numbers and values
// represented by a UniformQuantizedType.
// Note that this is not expected to be speedy and may be superseded eventually
// by a more optimal implementation.
// Also, the interface assumes that quantization is done per-layer and will
// need to be wider for various per-channel schemes. As such, this is a
// placeholder.
class UniformQuantizedValueConverter {
 public:
  explicit UniformQuantizedValueConverter(
      quant::UniformQuantizedType uniform_type)
      : UniformQuantizedValueConverter(
            uniform_type.getScale(),
            static_cast<double>(uniform_type.getZeroPoint()),
            static_cast<double>(uniform_type.getStorageTypeMin()),
            static_cast<double>(uniform_type.getStorageTypeMax()),
            uniform_type.getStorageTypeIntegralWidth(),
            uniform_type.isSigned()) {
    assert(isa<FloatType>(uniform_type.getExpressedType()));
    assert(uniform_type.getStorageType().isSignlessInteger());
  }

  UniformQuantizedValueConverter(double scale, double zero_point,
                                 double clamp_min, double clamp_max,
                                 uint32_t storage_bit_width, bool is_signed)
      : scale_(scale),
        zero_point_(zero_point),
        clamp_min_(clamp_min),
        clamp_max_(clamp_max),
        scale_double_(scale),
        zero_point_double_(zero_point),
        clamp_min_double_(clamp_min),
        clamp_max_double_(clamp_max),
        storage_bit_width_(storage_bit_width),
        is_signed_(is_signed),
        round_mode_(APFloat::rmNearestTiesToAway) {}

  UniformQuantizedValueConverter(double scale, double zero_point,
                                 const APFloat& clamp_min,
                                 const APFloat& clamp_max,
                                 uint32_t storage_bit_width, bool is_signed)
      : scale_(scale),
        zero_point_(zero_point),
        clamp_min_(clamp_min),
        clamp_max_(clamp_max),
        scale_double_(scale),
        zero_point_double_(zero_point),
        clamp_min_double_(clamp_min.convertToDouble()),
        clamp_max_double_(clamp_max.convertToDouble()),
        storage_bit_width_(storage_bit_width),
        is_signed_(is_signed),
        round_mode_(APFloat::rmNearestTiesToAway) {}

  virtual APInt quantizeFloatToInt(APFloat expressed_value) const {
    // This function is a performance critical code path in quantization
    // since it runs for each single float parameter value.

    // Specialize f32->u8/i8 case to optimize performance.
    if (&expressed_value.getSemantics() == &APFloat::IEEEsingle() &&
        storage_bit_width_ == 8 &&
        round_mode_ == llvm::APFloatBase::rmNearestTiesToAway) {
      return quantizeF32ToInt8(expressed_value);
    }

    bool lossy;
    expressed_value.convert(scale_.getSemantics(), round_mode_, &lossy);
    // fixed_point = clamp(clamp_min, clamp_max, (
    //   roundHalfToEven(expressed / scale) + zero_point))
    APFloat scaled = (expressed_value / scale_);
    scaled.roundToIntegral(round_mode_);
    scaled.add(zero_point_, round_mode_);
    APFloat fixed_point = llvm::minimum(scaled, clamp_max_);
    fixed_point = llvm::maximum(fixed_point, clamp_min_);

    llvm::APSInt result(storage_bit_width_, !is_signed_);
    fixed_point.convertToInteger(result, round_mode_, &lossy);

    return std::move(result);
  }

  int64_t quantizeFloatToInt64(APFloat expressed_value) const {
    const APInt q_value = quantizeFloatToInt(std::move(expressed_value));
    return is_signed_ ? q_value.getSExtValue() : q_value.getZExtValue();
  }

  virtual ~UniformQuantizedValueConverter() = default;

 private:
  // An optimized implementation to quantize f32 to i8/u8 with C++ native
  // arithmetic.
  virtual APInt quantizeF32ToInt8(const APFloat& expressed_value) const {
    assert(&expressed_value.getSemantics() == &APFloat::IEEEsingle());
    assert(storage_bit_width_ == 8);
    assert(round_mode_ == llvm::APFloatBase::rmNearestTiesToAway);

    const float real_value = expressed_value.convertToFloat();

    const double scaled = real_value / scale_double_ + zero_point_double_;
    // Round to nearest integer with halfway cases rounded away from zero.
    const double scaled_rounded = std::round(scaled);
    const double clamped = std::min(std::max(scaled_rounded, clamp_min_double_),
                                    clamp_max_double_);

    uint64_t signless_result;
    if (is_signed_) {
      int64_t clamped_int = static_cast<int8_t>(clamped);
      memcpy(&signless_result, &clamped_int, sizeof(clamped_int));
    } else {
      signless_result = static_cast<uint8_t>(clamped);
    }
    return APInt(storage_bit_width_, signless_result, /*isSigned=*/is_signed_);
  }

  // Keep both APFloat and double versions of the quantization parameters
  // around since they will be used in generic and specialized arithmetic,
  // respectively.
  const APFloat scale_;
  const APFloat zero_point_;
  const APFloat clamp_min_;
  const APFloat clamp_max_;

  const double scale_double_;
  const double zero_point_double_;
  const double clamp_min_double_;
  const double clamp_max_double_;

  const uint32_t storage_bit_width_;
  const bool is_signed_;
  const llvm::APFloat::roundingMode round_mode_;
};

// An utility class to quantize an attribute by the per-axis quantization
// parameters. The size of the quantization dim in the converted elements
// attribute should match the size of of scales/zero_points vectors in the
// quantization parameters.
class UniformQuantizedPerAxisValueConverter {
 public:
  explicit UniformQuantizedPerAxisValueConverter(
      quant::UniformQuantizedPerAxisType uniform_type)
      : scales_(uniform_type.getScales()),
        zero_points_(uniform_type.getZeroPoints()),
        clamp_min_(static_cast<double>(uniform_type.getStorageTypeMin())),
        clamp_max_(static_cast<double>(uniform_type.getStorageTypeMax())),
        storage_bit_width_(uniform_type.getStorageTypeIntegralWidth()),
        is_signed_(uniform_type.isSigned()),
        quantization_dim_(uniform_type.getQuantizedDimension()) {
    assert(isa<FloatType>(uniform_type.getExpressedType()));
    assert(uniform_type.getStorageType().isSignlessInteger());
    assert(scales_.size() == zero_points_.size());
  }

  // Quantize an Attribute by the quantization parameters. Return nullptr if
  // the conversion fails or the input array isn't an ElementsAttr.
  ElementsAttr convert(Attribute real_value);

 private:
  // Quantize an DenseFPElementsAttr by the quantization parameters.
  DenseElementsAttr convert(DenseFPElementsAttr attr);

  // Get a uniform converter for the index-th chunk along the quantizationDim.
  // All the elements in this chunk is quantized by the returned converter.
  UniformQuantizedValueConverter getPerChunkConverter(int index) const {
    return UniformQuantizedValueConverter(scales_[index], zero_points_[index],
                                          clamp_min_, clamp_max_,
                                          storage_bit_width_, is_signed_);
  }

  const ArrayRef<double> scales_;
  const ArrayRef<int64_t> zero_points_;
  const APFloat clamp_min_;
  const APFloat clamp_max_;
  const uint32_t storage_bit_width_;
  const bool is_signed_;
  int32_t quantization_dim_;
};

}  // namespace mlir::quantfork

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_COMMON_IR_UNIFORMSUPPORT_H_
