/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_PASSES_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla {
namespace mlir_gpu {

// TODO(herhut, pifon): Move these passes to MLIR Core.

/// Replaces a FusionOp by the operations contained in its region.
std::unique_ptr<mlir::FunctionPass> createFusionOpRemoverPass();

/// Replaces a load that immediately follows a store to the same address with
/// the stored value. This needs generalization.
std::unique_ptr<mlir::FunctionPass> createStoreForwardingPass();

/// Removes temporary buffers that are only written to but never read from or
/// that are read but the read value is not used. Needs an analysis that proves
/// that loads and stores are side-effect free (in bounds, no aliasing, etc.).
std::unique_ptr<mlir::FunctionPass> createDeadTempBufferRemovalPass();

/// Sorts the operands to the kernel for a deterministic order. First operands
/// that are defined by function arguments, followed by operands that are
/// returned from the function. This only works for simple functions without
/// control flow and can be used in cases where the kernel is extracted and used
/// independently of the host-side code.
std::unique_ptr<mlir::FunctionPass> createRewriteKernelSignaturePass();

/// We need to direct fusion to the inner loops. This cannot be done with
/// a passmanager alone ATM, as nested pass managers require operations to
/// be closed from above.
std::unique_ptr<mlir::FunctionPass> createFuseInnerParallelLoopsPass();

/// Greedily maps loops to GPU hardware dimensions.
std::unique_ptr<mlir::FunctionPass> createMapParallelLoopsPass();

/// Collapses all loop dimension into the first one.
std::unique_ptr<mlir::FunctionPass>
createParallelLoopCollapsingToFirstDimPass();

#define GEN_PASS_REGISTRATION
#include "tensorflow/compiler/xla/service/mlir_gpu/passes.h.inc"

}  // namespace mlir_gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_MLIR_GPU_PASSES_H_
