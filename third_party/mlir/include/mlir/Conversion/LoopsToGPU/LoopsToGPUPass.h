//===- LoopsToGPUPass.h - Pass converting loops to GPU kernels --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPUPASS_H_
#define MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPUPASS_H_

#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
class FuncOp;
template <typename T> class OpPassBase;

/// Create a pass that converts loop nests into GPU kernels.  It considers
/// top-level affine.for and linalg.for operations as roots of loop nests and
/// converts them to the gpu.launch operations if possible.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
std::unique_ptr<OpPassBase<FuncOp>>
createSimpleLoopsToGPUPass(unsigned numBlockDims, unsigned numThreadDims);

/// Create a pass that converts every loop operation within the body of the
/// FuncOp into a GPU launch. The number of workgroups and workgroup size for
/// the implementation is controlled by SSA values passed into conversion
/// method. For testing, the values are set as constants obtained from a command
/// line flag. See convertLoopToGPULaunch for a description of the required
/// semantics of the converted loop operation.
std::unique_ptr<OpPassBase<FuncOp>>
createLoopToGPUPass(ArrayRef<int64_t> numWorkGroups,
                    ArrayRef<int64_t> workGroupSize);
} // namespace mlir

#endif // MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPUPASS_H_
