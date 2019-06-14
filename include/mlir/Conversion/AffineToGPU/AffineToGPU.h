//===- AffineToGPU.h - Convert an affine loops to GPU kernels ---*- C++ -*-===//
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
#ifndef MLIR_CONVERSION_AFFINETOGPU_AFFINETOGPU_H_
#define MLIR_CONVERSION_AFFINETOGPU_AFFINETOGPU_H_

namespace mlir {
class AffineForOp;
struct LogicalResult;

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
} // namespace mlir

#endif // MLIR_CONVERSION_AFFINETOGPU_AFFINETOGPU_H_
