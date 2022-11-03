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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_COMPILATION_PIPELINE_CPU_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_COMPILATION_PIPELINE_CPU_H_

#include <functional>

#include "tensorflow/compiler/xla/mlir/runtime/transforms/compilation_pipeline_options.h"
#include "tensorflow/compiler/xla/runtime/compiler.h"

namespace xla {
namespace runtime {

struct CpuPipelineOptions {
  CompilationPipelineOptions common_options;

  // Byte alignment for allocated memrefs. Depending on the compiler flags
  // Tensorflow requires tensors to be aligned on 16, 32 or 64 bytes.
  int alignment = 0;

  // Enables math approximations that emit AVX2 intrinsics.
#ifdef __AVX2__
  bool math_avx2 = true;
#else
  bool math_avx2 = false;
#endif
};

// Registers dialects, interfaces and dialects translations with the registry
// required by the default XLA-CPU runtime compilation pipeline.
void RegisterDefaultXlaCpuRuntimeDialects(DialectRegistry& dialects);

// Creates default XLA-CPU runtime compilation pipeline that lowers from the
// `rt` and `memref` dialects to the LLVMIR dialect. This is a very simple
// pipeline that is mostly intended for writing tests for the XLA runtime, and
// it is expected that all end users will construct their own compilation
// pipelines from the available XLA and MLIR passes.
void CreateDefaultXlaCpuRuntimeCompilationPipeline(
    PassManager& passes, const CpuPipelineOptions& opts);

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_RUNTIME_TRANSFORMS_COMPILATION_PIPELINE_CPU_H_
