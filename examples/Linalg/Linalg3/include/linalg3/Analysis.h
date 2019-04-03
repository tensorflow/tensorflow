//===- Analysis.h - Linalg dialect Analysis function definitions ----------===//
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

#ifndef LINALG3_ANALYSIS_H_
#define LINALG3_ANALYSIS_H_

#include "linalg2/Analysis.h"

namespace mlir {
class AffineMap;
} // namespace mlir

namespace linalg {

/// Given a `map` specification and a subset of its results
/// `[beginResult, endResult)`, returns the inverse map that maps result
/// positions to dim positions.
mlir::AffineMap inverseSubMap(mlir::AffineMap map, unsigned beginResult = 0,
                              unsigned endResult = 0);

} // namespace linalg

#endif // LINALG3_ANALYSIS_H_
