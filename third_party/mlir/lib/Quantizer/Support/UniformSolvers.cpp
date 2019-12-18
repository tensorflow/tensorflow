//===- UniformSolvers.cpp - Uniform type solver algorithms ----------------===//
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

#include "mlir/Quantizer/Support/UniformSolvers.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>

using namespace mlir;
using namespace mlir::quantizer;

bool UniformParamsFromMinMaxSolver::compute() {
  // Compute adjMin, adjMax, clamping to ensure that they straddle zero.
  if (boundingMin > 0 && boundingMax >= boundingMin) {
    // Lop-sided to the positive.
    adjMin = 0;
    adjMax = boundingMax;
  } else if (boundingMax < 0 && boundingMin <= boundingMax) {
    // Lop-sided to the negative.
    adjMin = boundingMin;
    adjMax = 0;
  } else if (boundingMin <= 0 && boundingMax >= 0) {
    adjMin = boundingMin;
    adjMax = boundingMax;
  } else {
    // Illegal bounds.
    return satisfied = false;
  }

  const double origMinAdj = adjMin;
  const double origMaxAdj = adjMax;
  const double numLevelsDouble = storageParams.numLevels;

  struct fns {
    static std::pair<double, double>
    computeMinMax(double boundingMin, double numLevels, double delta) {
      double adjMin = delta * std::floor(boundingMin / delta);
      return std::make_pair(adjMin, adjMin + numLevels * delta);
    }
    static double overshoot(double boundingMin, double boundingMax,
                            double numLevels, double delta) {
      auto adjMinMax = computeMinMax(boundingMin, numLevels, delta);
      double maxOvershoot = adjMinMax.second - boundingMax;
      double minOvershoot = boundingMin - adjMinMax.first;
      // If undershooting on the min or max end, return that because it is
      // to be unconditionally avoided. Otherwise return the end with the
      // greatest magnitude of overshoot.
      if (maxOvershoot < 0)
        return maxOvershoot;
      if (minOvershoot < 0)
        return minOvershoot;
      return std::max(maxOvershoot, minOvershoot);
    }
  };

  // Bisect to find a suitable delta, starting with bounds of deltaInit
  // and deltaMax.
  double deltaInit = (adjMax - adjMin) / numLevelsDouble;
  double deltaMax =
      ((numLevelsDouble * deltaInit) + 2 * deltaInit) / numLevelsDouble;
  double deltaMid;
  double prevDeltaMid = 0.0;
  for (stepCount = 0; stepCount < 60; ++stepCount) {
    deltaMid = (deltaInit + deltaMax) / 2.0;
    auto fInit =
        fns::overshoot(origMinAdj, origMaxAdj, numLevelsDouble, deltaInit);
    auto fMid =
        fns::overshoot(origMinAdj, origMaxAdj, numLevelsDouble, deltaMid);
    if (fMid == 0 || (fMid > 0 && std::fabs(deltaMid - prevDeltaMid) < 1e-15)) {
      // Solution found (or step size is infinitesimal and an overshoot).
      // Empirically, this seems to terminate around 30-50 steps or so.
      // This will find a zero point for exactly representable ranges and
      // will terminate on a small step size for inexact, biasing towards
      // overshooting.
      delta = deltaMid;
      break;
    }
    bool signMid = fMid > 0;
    bool signInit = fInit > 0;
    if (signMid == signInit) {
      deltaInit = deltaMid;
    } else {
      deltaMax = deltaMid;
    }
    prevDeltaMid = deltaMid;
  }
  delta = deltaMid;

  // Recalculate adjMin/adjMax based on new delta.
  auto adjMinMax = fns::computeMinMax(origMinAdj, numLevelsDouble, delta);
  adjMin = adjMinMax.first;
  adjMax = adjMinMax.second;

  satisfied = false;
  zp = 0;

  if (!std::isnan(delta) && !std::isnan(adjMin) && !std::isnan(adjMax)) {
    satisfied = true;
    // Finally, scale and zeroPoint. Since it casts to integer, only valid
    // if the inputs are valid.
    zp = std::round(storageParams.minValue - adjMin / delta);
  }

  return satisfied;
}

int64_t UniformParamsFromMinMaxSolver::quantize(double x) const {
  int64_t xq = std::round(x / delta + zp);
  return std::max<int64_t>(0, std::min<int64_t>(storageParams.numLevels, xq));
}

double UniformParamsFromMinMaxSolver::dequantize(int64_t xq) const {
  return (xq - zp) * delta;
}

namespace mlir {
namespace quantizer {

raw_ostream &operator<<(raw_ostream &os, const UniformStorageParams &p) {
  os << "UniformStorageParams{" << p.numLevels << ", " << p.minValue << "}";
  return os;
}

raw_ostream &operator<<(raw_ostream &os,
                        const UniformParamsFromMinMaxSolver &s) {
  os << "UniformParamsFromMinMaxSolver(" << s.getStepCount() << "){";
  os << "(" << s.getBoundingMin() << ":" << s.getBoundingMax() << ") -> ";
  if (!s.isSatisfied()) {
    os << "unsat}";
    return os;
  }

  os << "(" << s.getAdjMin() << ":" << s.getAdjMax() << ")";
  os << ", scale = " << s.getScale();
  os << ", zp = " << s.getZp();
  os << "}";

  return os;
}

} // end namespace quantizer
} // end namespace mlir
