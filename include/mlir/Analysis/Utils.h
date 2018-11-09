//===- Utils.h - General analysis utilities ---------------------*- C++ -*-===//
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
// This header file defines prototypes for various transformation utilities for
// memref's and non-loop IR structures. These are not passes by themselves but
// are used either by passes, optimization sequences, or in turn by other
// transformation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_UTILS_H
#define MLIR_ANALYSIS_UTILS_H

namespace mlir {

class FlatAffineConstraints;
class OperationStmt;
class Statement;

/// Returns true if statement 'a' dominates statement b.
bool dominates(const Statement &a, const Statement &b);

/// Returns true if statement 'a' properly dominates statement b.
bool properlyDominates(const Statement &a, const Statement &b);

/// Returns the memory region accessed by this memref.
bool getMemoryRegion(OperationStmt *opStmt, FlatAffineConstraints *region);

} // end namespace mlir

#endif // MLIR_ANALYSIS_UTILS_H
