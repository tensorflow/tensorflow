//===- LoopsToGPU.h - Convert loop nests to GPU kernels ---------*- C++ -*-===//
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
#ifndef MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPU_H_
#define MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPU_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineForOp;
struct LogicalResult;
class Value;

// TODO(riverriddle) Remove this after Value is value-typed.
using ValuePtr = Value *;

namespace loop {
class ForOp;
} // end namespace loop

/// Convert a perfect affine loop nest with the outermost loop identified by
/// `forOp` into a gpu::Launch operation.  Map `numBlockDims` outer loops to
/// GPU blocks and `numThreadDims` to GPU threads.  The bounds of the loops that
/// are mapped should be independent of the induction variables of the other
/// mapped loops.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
LogicalResult convertAffineLoopNestToGPULaunch(AffineForOp forOp,
                                               unsigned numBlockDims,
                                               unsigned numThreadDims);

/// Convert a perfect linalg loop nest with the outermost loop identified by
/// `forOp` into a gpu::Launch operation.  Map `numBlockDims` outer loops to
/// GPU blocks and `numThreadDims` to GPU threads.  The bounds of the loops that
/// are mapped should be independent of the induction variables of the other
/// mapped loops.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
LogicalResult convertLoopNestToGPULaunch(loop::ForOp forOp,
                                         unsigned numBlockDims,
                                         unsigned numThreadDims);

/// Convert a loop operation into a GPU launch with the values provided in
/// `numWorkGroups` as the grid size and the values provided in `workGroupSizes`
/// as the block size. Size of `numWorkGroups` and workGroupSizes` must be less
/// than or equal to 3. The loop operation can be an imperfectly nested
/// computation with the following restrictions:
/// 1) The loop nest must contain as many perfectly nested loops as the number
/// of values passed in through `numWorkGroups`. This corresponds to the number
/// of grid dimensions of the launch. All loops within the loop nest must be
/// parallel.
/// 2) The body of the innermost loop of the above perfectly nested loops, must
/// contain statements that satisfy one of the two conditions below:
///   a) A perfect loop nest of depth greater than or equal to the number of
///   values passed in through `workGroupSizes`, i.e. the number of thread
///   dimensions of the launch. Loops at depth less than or equal to size of
///   `workGroupSizes` must be parallel. Loops nested deeper can be sequential
///   and are retained as such in the generated GPU launch code.
///   b) Statements that are safe to be executed by all threads within the
///   workgroup. No checks are performed that this is indeed the case.
///   TODO(ravishankarm) : Add checks that verify 2(b) above.
/// The above conditions are assumed to be satisfied by the computation rooted
/// at `forOp`.
LogicalResult convertLoopToGPULaunch(loop::ForOp forOp,
                                     ArrayRef<ValuePtr> numWorkGroups,
                                     ArrayRef<ValuePtr> workGroupSizes);

} // namespace mlir

#endif // MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPU_H_
