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

#include "tensorflow/compiler/mlir/lite/quantization/ir/FakeQuantSupport.h"

#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project

using namespace mlir;
using namespace mlir::quantfork;

static bool getDefaultStorageParams(unsigned numBits, bool narrowRange,
                                    bool isSigned, MLIRContext *ctx,
                                    Type &storageType, int64_t &qmin,
                                    int64_t &qmax) {
  // Hard-coded type mapping from TFLite.
  if (numBits <= 4) {
    storageType = IntegerType::get(ctx, 4);
    if (isSigned) {
      qmin = -8;
      qmax = 7;
    } else {
      qmin = 0;
      qmax = 15;
    }
  } else if (numBits <= 8) {
    storageType = IntegerType::get(ctx, 8);
    if (isSigned) {
      qmin = -128;
      qmax = 127;
    } else {
      qmin = 0;
      qmax = 255;
    }
  } else if (numBits <= 16) {
    storageType = IntegerType::get(ctx, 16);
    if (isSigned) {
      qmin = -32768;
      qmax = 32767;
    } else {
      qmin = 0;
      qmax = 65535;
    }
  } else if (numBits <= 32) {
    storageType = IntegerType::get(ctx, 32);
    if (isSigned) {
      qmin = std::numeric_limits<int32_t>::min();
      qmax = std::numeric_limits<int32_t>::max();
    } else {
      qmin = std::numeric_limits<uint32_t>::min();
      qmax = std::numeric_limits<uint32_t>::max();
    }
  } else {
    return true;
  }

  // Handle narrowRange.
  if (narrowRange) {
    qmin += 1;
  }
  return false;
}

// This is a specific implementation of nudging:
// If 0.0 < rmin < rmax or rmin < rmax < 0.0, the range will be shifted
// to include 0.0, but the range width size (rmax-rmin) isn't changed. The zero
// point is derived from the shifted range, and the scale isn't changed. As
// a consequence some values, which are supposed in the original [rmin, rmax]
// range will be outside the shifted range and be clamped during quantization.
// TODO: we should nudge the scale as well, but that requires the
// fake quant op used in the training to use the nudged scale as well.
static void getNudgedScaleAndZeroPoint(int64_t qmin, int64_t qmax, double rmin,
                                       double rmax, double &scale,
                                       int64_t &nudgedZeroPoint) {
  // Determine the scale.
  const double qminDouble = qmin;
  const double qmaxDouble = qmax;
  scale = (rmax - rmin) / (qmaxDouble - qminDouble);

  // Zero point computation.
  // In float, solve the affine equation for any known pair
  // (real value, corresponding quantized value), of which, two such pairs
  // are known: (rmin, qmin), (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair will be
  // roughly machine_epsilon * (sum of absolute values of terms).
  // Use the variant that adds the smaller error.
  const double zeroPointFromMin = qminDouble - rmin / scale;
  const double zeroPointFromMinError =
      std::abs(qminDouble) + std::abs(rmin / scale);
  const double zeroPointFromMax = qmaxDouble - rmax / scale;
  const double zeroPointFromMaxError =
      std::abs(qmaxDouble) + std::abs(rmax / scale);

  const double zeroPointDouble = (zeroPointFromMinError < zeroPointFromMaxError)
                                     ? zeroPointFromMin
                                     : zeroPointFromMax;

  // Now nudge the zero point to be an integer.
  nudgedZeroPoint = 0;
  if (zeroPointDouble < qminDouble) {
    nudgedZeroPoint = qmin;
  } else if (zeroPointDouble > qmaxDouble) {
    nudgedZeroPoint = qmax;
  } else {
    nudgedZeroPoint = round(zeroPointDouble);
  }

  // By construction, the nudged zero point should always be in range.
  assert(nudgedZeroPoint >= qmin);
  assert(nudgedZeroPoint <= qmax);
}

quant::UniformQuantizedType mlir::quantfork::fakeQuantAttrsToType(
    Location loc, unsigned numBits, double rmin, double rmax, bool narrowRange,
    Type expressedType, bool isSigned) {
  MLIRContext *ctx = expressedType.getContext();
  unsigned flags = isSigned ? quant::QuantizationFlags::Signed : 0;
  Type storageType;
  int64_t qmin;
  int64_t qmax;
  if (getDefaultStorageParams(numBits, narrowRange, isSigned, ctx, storageType,
                              qmin, qmax)) {
    return (emitError(loc, "unsupported FakeQuant number of bits: ") << numBits,
            nullptr);
  }

  // Special case where min/max is close enough. The tensor contents are all
  // 0.0s, so the scale is set to 1.0 and the tensor can be quantized to zero
  // points and dequantized to 0.0.
  if (std::fabs(rmax - rmin) < std::numeric_limits<double>::epsilon()) {
    return quant::UniformQuantizedType::getChecked(
        loc, flags, storageType, expressedType, 1.0, qmin, qmin, qmax);
  }

  double scale;
  int64_t nudgedZeroPoint;
  getNudgedScaleAndZeroPoint(qmin, qmax, rmin, rmax, scale, nudgedZeroPoint);

  return quant::UniformQuantizedType::getChecked(loc, flags, storageType,
                                                 expressedType, scale,
                                                 nudgedZeroPoint, qmin, qmax);
}

quant::UniformQuantizedPerAxisType mlir::quantfork::fakeQuantAttrsToType(
    Location loc, unsigned numBits, int32_t quantizedDimension,
    ArrayRef<double> rmins, ArrayRef<double> rmaxs, bool narrowRange,
    Type expressedType, bool isSigned) {
  size_t axisSize = rmins.size();
  if (axisSize != rmaxs.size()) {
    return (emitError(loc, "mismatched per-axis min and max size: ")
                << axisSize << " vs. " << rmaxs.size(),
            nullptr);
  }

  MLIRContext *ctx = expressedType.getContext();
  Type storageType;
  int64_t qmin;
  int64_t qmax;
  if (getDefaultStorageParams(numBits, narrowRange, isSigned, ctx, storageType,
                              qmin, qmax)) {
    return (emitError(loc, "unsupported FakeQuant number of bits: ") << numBits,
            nullptr);
  }

  SmallVector<double, 4> scales;
  SmallVector<int64_t, 4> zeroPoints;
  scales.reserve(axisSize);
  zeroPoints.reserve(axisSize);
  for (size_t axis = 0; axis != axisSize; ++axis) {
    double rmin = rmins[axis];
    double rmax = rmaxs[axis];
    if (std::fabs(rmax - rmin) < std::numeric_limits<double>::epsilon()) {
      scales.push_back(1.0);
      zeroPoints.push_back(qmin);
      continue;
    }

    double scale;
    int64_t nudgedZeroPoint;
    getNudgedScaleAndZeroPoint(qmin, qmax, rmin, rmax, scale, nudgedZeroPoint);
    scales.push_back(scale);
    zeroPoints.push_back(nudgedZeroPoint);
  }

  unsigned flags = isSigned ? quant::QuantizationFlags::Signed : 0;
  return quant::UniformQuantizedPerAxisType::getChecked(
      loc, flags, storageType, expressedType, scales, zeroPoints,
      quantizedDimension, qmin, qmax);
}
