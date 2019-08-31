//===- LoopsToGPUPass.h - Pass converting loops to GPU kernels --*- C++ -*-===//
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
#ifndef MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPUPASS_H_
#define MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPUPASS_H_

#include <memory>

namespace mlir {
class FuncOp;
template <typename T> class OpPassBase;
using FunctionPassBase = OpPassBase<FuncOp>;

/// Create a pass that converts loop nests into GPU kernels.  It considers
/// top-level affine.for and linalg.for operations as roots of loop nests and
/// converts them to the gpu.launch operations if possible.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
std::unique_ptr<FunctionPassBase>
createSimpleLoopsToGPUPass(unsigned numBlockDims, unsigned numThreadDims);
} // namespace mlir

#endif // MLIR_CONVERSION_LOOPSTOGPU_LOOPSTOGPUPASS_H_
