//===- FakeQuantSupport.cpp - Support utilities for FakeQuant ops ---------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Dialect/QuantOps/FakeQuantSupport.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"

using namespace mlir;
using namespace mlir::quant;

UniformQuantizedType
mlir::quant::fakeQuantAttrsToType(Location loc, unsigned numBits, double rmin,
                                  double rmax, bool narrowRange,
                                  Type expressedType, bool isSigned) {
  MLIRContext *ctx = expressedType.getContext();
  Type storageType;
  unsigned flags;
  int64_t qmin;
  int64_t qmax;

  // Hard-coded type mapping from TFLite.
  if (numBits <= 8) {
    storageType = IntegerType::get(8, ctx);
    if (isSigned) {
      flags = QuantizationFlags::Signed;
      qmin = -128;
      qmax = 127;
    } else {
      flags = 0;
      qmin = 0;
      qmax = 255;
    }
  } else if (numBits <= 16) {
    storageType = IntegerType::get(16, ctx);
    if (isSigned) {
      flags = QuantizationFlags::Signed;
      qmin = -32768;
      qmax = 32767;
    } else {
      flags = 0;
      qmin = 0;
      qmax = 65535;
    }
  } else {
    emitError(loc, "unsupported FakeQuant number of bits: ") << numBits;
    return nullptr;
  }

  // Handle narrowRange.
  if (narrowRange) {
    qmin += 1;
  }

  // Range must straddle zero.
  if (rmin > 0.0 || rmax < 0.0) {
    return (emitError(loc, "FakeQuant range must straddle zero: [")
                << rmin << "," << rmax << "]",
            nullptr);
  }

  // Special case where min/max is a point. Must be 0.
  if (rmin == rmax) {
    return UniformQuantizedType::getChecked(flags, storageType, expressedType,
                                            0.0, 0, qmin, qmax, loc);
  }

  // Determine the scale.
  const double qminDouble = qmin;
  const double qmaxDouble = qmax;
  const double scale = (rmax - rmin) / (qmaxDouble - qminDouble);

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
  int64_t nudgedZeroPoint = 0;
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

  return UniformQuantizedType::getChecked(flags, storageType, expressedType,
                                          scale, nudgedZeroPoint, qmin, qmax,
                                          loc);
}
