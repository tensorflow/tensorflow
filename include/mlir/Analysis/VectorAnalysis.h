//===- VectorAnalysis.h - Analysis for Vectorization -------*- C++ -*-=======//
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

#ifndef MLIR_ANALYSIS_VECTORANALYSIS_H_
#define MLIR_ANALYSIS_VECTORANALYSIS_H_

#include "mlir/Support/LLVM.h"

namespace mlir {

class AffineMap;
class MemRefType;
class OperationStmt;
class VectorType;

/// Computes and returns the multi-dimensional ratio of `superShape` to
/// `subShape`. This is calculated by performing a traversal from minor to major
/// dimensions (i.e. in reverse shape order). If integral division is not
/// possible, returns None.
///
/// Examples:
///   - shapeRatio({3, 4, 5, 8}, {2, 5, 2}) returns {3, 2, 1, 4}
///   - shapeRatio({3, 4, 4, 8}, {2, 5, 2}) returns None
///   - shapeRatio({1, 2, 10, 32}, {2, 5, 2}) returns {1, 1, 2, 16}
llvm::Optional<llvm::SmallVector<unsigned, 4>>
shapeRatio(ArrayRef<int> superShape, ArrayRef<int> subShape);

/// Computes and returns the multi-dimensional ratio of the shapes of
/// `superVecto` to `subVector`. If integral division is not possible, returns
/// None.
llvm::Optional<llvm::SmallVector<unsigned, 4>>
shapeRatio(VectorType superVectorType, VectorType subVectorType);

/// Creates a permutation map to be used as an attribute in VectorTransfer ops.
/// Currently only returns the minor vectorType.rank identity submatrix.
///
/// For example, assume memrefType is of rank 5 and vectorType is of rank 3,
/// returns the affine map:
///     (d0, d1, d2, d3, d4) -> (d2, d3, d4)
///
/// TODO(ntv): support real permutations.
AffineMap makePermutationMap(MemRefType memrefType, VectorType vectorType);

namespace matcher {

/// Matches vector_transfer_read, vector_transfer_write and ops that return a
/// vector type that is at least a 2-multiple of the sub-vector type. This
/// allows passing over other smaller vector types in the function and avoids
/// interfering with operations on those.
/// This is a first approximation, it can easily be extended in the future.
/// TODO(ntv): this could all be much simpler if we added a bit that a vector
/// type to mark that a vector is a strict super-vector but it still does not
/// warrant adding even 1 extra bit in the IR for now.
bool operatesOnStrictSuperVectors(const OperationStmt &stmt,
                                  VectorType subVectorType);

} // end namespace matcher
} // end namespace mlir

#endif // MLIR_ANALYSIS_VECTORANALYSIS_H_
