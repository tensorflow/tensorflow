/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_RUNTIME_COMPILATION_PIPELINE_GPU_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_RUNTIME_COMPILATION_PIPELINE_GPU_H_

#include <functional>

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/transforms/runtime/compilation_pipeline_options.h"

namespace xla {
namespace runtime {

// Registers dialects, interfaces and dialects translations with the registry
// required by the default XLA-GPU runtime compilation pipeline.
void RegisterDefaultXlaGpuRuntimeDialects(mlir::DialectRegistry& registry);

// Creates default XLA-GPU runtime compilation pipeline that lowers from the
// `rt` and `memref` dialects to the LLVMIR dialect. This is a very simple
// pipeline that is mostly intended for writing tests for the XLA runtime, and
// it is expected that all end users will construct their own compilation
// pipelines from the available XLA and MLIR passes.
void CreateDefaultXlaGpuRuntimeCompilationPipeline(
    mlir::OpPassManager& pm, const CompilationPipelineOptions& opts);

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_RUNTIME_COMPILATION_PIPELINE_GPU_H_
