//===- UniformSolvers.h - Uniform type solver algorithms --------*- C++ -*-===//
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
//
// This file defines algorithms for solving uniform type parameters for various
// conditions (i.e. fixed-point, affine, scale matching, etc).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_SUPPORT_UNIFORMSOLVERS_H
#define MLIR_QUANTIZER_SUPPORT_UNIFORMSOLVERS_H

#include <cstdint>
#include <limits>

namespace llvm {
class raw_ostream;
} // end namespace llvm

namespace mlir {
namespace quantizer {

struct UniformStorageParams {
  static UniformStorageParams getQuint8() { return {255, 0}; }
  static UniformStorageParams getQuint8SymmetricRight() { return {254, 1}; }
  static UniformStorageParams getQuint16() { return {32767, 0}; }

  uint64_t numLevels;
  int64_t minValue;
};

/// Solves for the uniform quantization scheme parameters delta and z given
/// bounding min/max.
class UniformParamsFromMinMaxSolver {
public:
  UniformParamsFromMinMaxSolver(const UniformStorageParams &storageParams,
                                double boundingMin, double boundingMax)
      : storageParams(storageParams), boundingMin(boundingMin),
        boundingMax(boundingMax) {}

  /// Performs the computation, returning whether satisfied.
  bool compute();

  // Params.
  double getBoundingMin() const { return boundingMin; }
  double getBoundingMax() const { return boundingMax; }
  bool isSatisfied() const { return satisfied; }
  double getAdjMin() const { return adjMin; }
  double getAdjMax() const { return adjMax; }
  double getScale() const { return delta; }
  int64_t getZp() const { return zp; }
  int getStepCount() const { return stepCount; }

  // Quantize and dequantize.
  int64_t quantize(double x) const;
  double dequantize(int64_t xq) const;

private:
  const UniformStorageParams storageParams;
  const double boundingMin;
  const double boundingMax;

  // Results
  int stepCount = 0;
  double adjMin = std::numeric_limits<double>::quiet_NaN();
  double adjMax = std::numeric_limits<double>::quiet_NaN();
  double delta = std::numeric_limits<double>::quiet_NaN();
  int64_t zp = 0;

  bool satisfied = false;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const UniformStorageParams &p);

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const UniformParamsFromMinMaxSolver &s);

} // end namespace quantizer
} // end namespace mlir

#endif // MLIR_QUANTIZER_SUPPORT_UNIFORMSOLVERS_H
