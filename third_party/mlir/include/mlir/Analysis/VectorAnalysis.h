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

#include "llvm/ADT/DenseMap.h"

namespace mlir {

class AffineApplyOp;
class AffineForOp;
class AffineMap;
class Location;
class MemRefType;
class OpBuilder;
class Operation;
class Value;
class VectorType;

/// Computes and returns the multi-dimensional ratio of `superShape` to
/// `subShape`. This is calculated by performing a traversal from minor to major
/// dimensions (i.e. in reverse shape order). If integral division is not
/// possible, returns None.
/// The ArrayRefs are assumed (and enforced) to only contain > 1 values.
/// This constraint comes from the fact that they are meant to be used with
/// VectorTypes, for which the property holds by construction.
///
/// Examples:
///   - shapeRatio({3, 4, 5, 8}, {2, 5, 2}) returns {3, 2, 1, 4}
///   - shapeRatio({3, 4, 4, 8}, {2, 5, 2}) returns None
///   - shapeRatio({1, 2, 10, 32}, {2, 5, 2}) returns {1, 1, 2, 16}
llvm::Optional<llvm::SmallVector<int64_t, 4>>
shapeRatio(ArrayRef<int64_t> superShape, ArrayRef<int64_t> subShape);

/// Computes and returns the multi-dimensional ratio of the shapes of
/// `superVector` to `subVector`. If integral division is not possible, returns
/// None.
/// Assumes and enforces that the VectorTypes have the same elemental type.
llvm::Optional<llvm::SmallVector<int64_t, 4>>
shapeRatio(VectorType superVectorType, VectorType subVectorType);

/// Constructs a permutation map of invariant memref indices to vector
/// dimension.
///
/// If no index is found to be invariant, 0 is added to the permutation_map and
/// corresponds to a vector broadcast along that dimension.
///
/// The implementation uses the knowledge of the mapping of loops to
/// vector dimension. `loopToVectorDim` carries this information as a map with:
///   - keys representing "vectorized enclosing loops";
///   - values representing the corresponding vector dimension.
/// Note that loopToVectorDim is a whole function map from which only enclosing
/// loop information is extracted.
///
/// Prerequisites: `opInst` is a vectorizable load or store operation (i.e. at
/// most one invariant index along each AffineForOp of `loopToVectorDim`).
///
/// Example 1:
/// The following MLIR snippet:
///
/// ```mlir
///    affine.for %i3 = 0 to %0 {
///      affine.for %i4 = 0 to %1 {
///        affine.for %i5 = 0 to %2 {
///          %a5 = load %arg0[%i4, %i5, %i3] : memref<?x?x?xf32>
///    }}}
/// ```
///
/// may vectorize with {permutation_map: (d0, d1, d2) -> (d2, d1)} into:
///
/// ```mlir
///    affine.for %i3 = 0 to %0 step 32 {
///      affine.for %i4 = 0 to %1 {
///        affine.for %i5 = 0 to %2 step 256 {
///          %4 = vector.transfer_read %arg0, %i4, %i5, %i3
///               {permutation_map: (d0, d1, d2) -> (d2, d1)} :
///               (memref<?x?x?xf32>, index, index) -> vector<32x256xf32>
///    }}}
/// ```
///
/// Meaning that vector.transfer_read will be responsible for reading the slice:
/// `%arg0[%i4, %i5:%15+256, %i3:%i3+32]` into vector<32x256xf32>.
///
/// Example 2:
/// The following MLIR snippet:
///
/// ```mlir
///    %cst0 = constant 0 : index
///    affine.for %i0 = 0 to %0 {
///      %a0 = load %arg0[%cst0, %cst0] : memref<?x?xf32>
///    }
/// ```
///
/// may vectorize with {permutation_map: (d0) -> (0)} into:
///
/// ```mlir
///    affine.for %i0 = 0 to %0 step 128 {
///      %3 = vector.transfer_read %arg0, %c0_0, %c0_0
///           {permutation_map: (d0, d1) -> (0)} :
///           (memref<?x?xf32>, index, index) -> vector<128xf32>
///    }
/// ````
///
/// Meaning that vector.transfer_read will be responsible of reading the slice
/// `%arg0[%c0, %c0]` into vector<128xf32> which needs a 1-D vector broadcast.
///
AffineMap makePermutationMap(
    Operation *op, ArrayRef<Value *> indices,
    const llvm::DenseMap<Operation *, unsigned> &loopToVectorDim);

namespace matcher {

/// Matches vector.transfer_read, vector.transfer_write and ops that return a
/// vector type that is a multiple of the sub-vector type. This allows passing
/// over other smaller vector types in the function and avoids interfering with
/// operations on those.
/// This is a first approximation, it can easily be extended in the future.
/// TODO(ntv): this could all be much simpler if we added a bit that a vector
/// type to mark that a vector is a strict super-vector but it still does not
/// warrant adding even 1 extra bit in the IR for now.
bool operatesOnSuperVectorsOf(Operation &op, VectorType subVectorType);

} // end namespace matcher
} // end namespace mlir

#endif // MLIR_ANALYSIS_VECTORANALYSIS_H_
