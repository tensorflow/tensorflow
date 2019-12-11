//===- VectorTransforms.h - Vector transformations as patterns --*- C++ -*-===//
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

#ifndef DIALECT_VECTOROPS_VECTORTRANSFORMS_H_
#define DIALECT_VECTOROPS_VECTORTRANSFORMS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class OwningRewritePatternList;

/// Collect a set of patterns to convert from the Vector dialect to itself.
/// Should be merged with populateVectorToAffineLoopsConversionPatterns.
void populateVectorToVectorConversionPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns,
    ArrayRef<int64_t> coarseVectorShape = {},
    ArrayRef<int64_t> fineVectorShape = {});

////////////////////////////////////////////////////////////////////////////////
// The following Declarative Rewrite Rule (DRR) helpers are used in rewrite
// patterns. As such, they must not call into `rewriter.erase/replace` APIs and
// it is the responsibility of the enclosing PatternRewriter to erase on
// success.
////////////////////////////////////////////////////////////////////////////////

namespace vector {

// Entry point for unrolling declarative pattern rewrites.
// `op` is unrolled to the `targetShape` as follows, for each of its operands:
//   1. the unrolled type `unrolledVectorType` and number of unrolled instances
//   `numUnrolledInstances` are computed from the `targetShape`. For now it is
//   assumed the unrolling factors divide the vector sizes.
//   2. a fakeFork cast op is inserted that takes the operand and returns
//   `numUnrolledInstances` results of type `unrolledVectorType`.
//   3. the original op is cloned `numUnrolledInstances` times, once for each
//   result of the fakeFork cast op.
//   4. a fakeJoin cast op takes all these results and merges them into a single
//   aggregate vector result whose size matches the original non-unrolled op
//   operand types.
//
// Example:
//
//    opA(operand0, operand1)  // numUnrolledInstances = 3
//
//            operand0                   operand1
//               |                          |
//             fork                       fork
//        <----------gather all fork ops --------->
//              /|\                        /|\
//          f00 f01 f02                f10 f11 f12
//        <---------- clone op 3 times --------->
//          opA0(f00, f10), opA1(f01, f11), opA2(f02, f12)
//                 \            |            /
//      <-------------------- join ------------------------->
//
// Other local patterns then kick in iteratively (including DCE) and compose
// until all the fakeFork and fakeJoin ops are removed.
//
// This will be extended in the future to support more advanced use cases than
// simple pointwise ops.
Value *unrollSingleResultOpMatchingType(PatternRewriter &builder, Operation *op,
                                        ArrayRef<int64_t> targetShape);

} // namespace vector
} // namespace mlir

#endif // DIALECT_VECTOROPS_VECTORTRANSFORMS_H_
