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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_CONVERT_COMPILED_OPS_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_CONVERT_COMPILED_OPS_H_

#include "iree-dialects/Dialect/Input/InputOps.h"
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/de_bufferization.h"
#include "tensorflow/compiler/xla/mlir/backends/gpu2/conversion/xla_gpu_api.h"

namespace xla {
namespace gpu {

// Forward declare.
class ThunkSequence;

// Appends patterns to convert LMHLO operations compiled to kernel thunks to
// IREEInput executable export and dispatch operations.
void populateCompiledOpsConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::TypeConverter &converter,
    mlir::iree_compiler::IREE::Input::ExecutableSourceOp executable_source,
    ThunkSequence *thunk_sequence, DeBufferization &state);

// Appends patterns to convert LMHLO operations compiled to kernel thunks to
// XLA:GPU runtime API calls.
void populateCompiledOpsConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::TypeConverter &converter,
                                           ThunkSequence *thunk_sequence,
                                           DeBufferization &state,
                                           XlaGpuApi &api,
                                           XlaGpuGraphs &graphs);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_GPU2_CONVERSION_CONVERT_COMPILED_OPS_H_
