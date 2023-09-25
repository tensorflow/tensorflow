/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_BACKENDS_GPU2_TRANSFORMS_PASSES_H_
#define XLA_MLIR_BACKENDS_GPU2_TRANSFORMS_PASSES_H_

namespace xla::gpu {

class ThunkSequence;  // forward declare

struct Gpu2PipelineOpts {};

}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
// TODO(ezhulenev): We currently do not build with XLA:GPU experimental runtime
// in open source because we do not have bazel dependency from XLA to IREE.
#if XLA_DISABLE_GPU2_COMPILER
//===----------------------------------------------------------------------===//

namespace mlir {
class OpPassManager;
}  // namespace mlir

namespace xla::gpu {
inline void populateGpu2RuntimePasses(mlir::OpPassManager&, ThunkSequence*,
                                      const Gpu2PipelineOpts& opts) {}
inline void registerGpu2Pases() {}
}  // namespace xla::gpu

//===----------------------------------------------------------------------===//
#else  // !XLA_DISABLE_GPU2_COMPILER
//===----------------------------------------------------------------------===//

#include <memory>
#include <optional>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla::gpu {

// Populate passes that lower MLIR modules from a combination of LMHLO and
// LMHLO_GPU dialects to the XLA:GPU runtime (aka IREE input dialects + XLA:GPU
// custom calls implementing library integration).
void populateGpu2RuntimePasses(mlir::OpPassManager& pm,
                               ThunkSequence* thunk_sequence,
                               const Gpu2PipelineOpts& opts);

//===----------------------------------------------------------------------===//
// Conversion from LMHLO dialects to XLA:GPU runtime
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp> >
createConvertToGpu2RuntimePass(ThunkSequence* thunk_sequence = nullptr);

//===----------------------------------------------------------------------===//
// XLA:GPU passes registration
//===----------------------------------------------------------------------===//

void registerGpu2Pases();

}  // namespace xla::gpu

#endif  // !XLA_DISABLE_GPU2_RUNTIME
#endif  // XLA_MLIR_BACKENDS_GPU2_TRANSFORMS_PASSES_H_
