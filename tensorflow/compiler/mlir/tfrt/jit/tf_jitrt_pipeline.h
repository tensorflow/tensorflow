/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_PIPELINE_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/Hashing.h"

namespace tensorflow {

struct TfJitRtPipelineOptions
    : public mlir::PassPipelineOptions<TfJitRtPipelineOptions> {
  Option<bool> vectorize{*this, "vectorize",
                         llvm::cl::desc("Enable tiling for vectorization."),
                         llvm::cl::init(false)};

  Option<bool> peel{*this, "peel", llvm::cl::desc("Enable loop peeling."),
                    llvm::cl::init(true)};

  Option<bool> fuse_fill{
      *this, "fuse-fill",
      llvm::cl::desc("Enable fusion of `linalg.fill` into a tiled reduction."),
      llvm::cl::init(true)};

  Option<int64_t> vector_size{*this, "vector-size",
                              llvm::cl::desc("Vector size for a 1D reduction."),
                              llvm::cl::init(8)};

  Option<int64_t> reduction_1d_tile_size{
      *this, "reduction-1d-tile-size",
      llvm::cl::desc("Tile size for a 1D reduction."), llvm::cl::init(32)};

  ListOption<int64_t> reduction_2d_tile_sizes{
      *this, "reduction-2d-tile-sizes",
      llvm::cl::desc("Tile sizes for a 2D reduction."), llvm::cl::ZeroOrMore};

  Option<bool> legalize_i1_tensors{
      *this, "legalize-i1-tensors",
      llvm::cl::desc("Convert i1 tensors to i8 tensors."),
      llvm::cl::init(false)};

  Option<bool> codegen_transpose{
      *this, "codegen-transpose",
      llvm::cl::desc(
          "Enable the specific code generation for transpose operations."),
      llvm::cl::init(false)};
};

// Make TfJitRtPipelineOptions hashable.
inline ::llvm::hash_code hash_value(const TfJitRtPipelineOptions& opts) {
  return ::llvm::hash_value(static_cast<bool>(opts.vectorize));
}

// Creates a pipeline that lowers modules from the Tensorflow dialect to
// the Linalg on buffers. `TfJitRtPipelineOptions` contains flags to
// enable/disable experimental features.
void CreateTfJitRtPipeline(mlir::OpPassManager& pm,
                           const TfJitRtPipelineOptions& options);

// Calls CreateTfJitRtPipeline with the default TfJitRtPipelineOptions.
void CreateDefaultTfJitRtPipeline(mlir::OpPassManager& pm);

// Creates a pipeline that runs on compiled module specialization. It runs the
// Tensorflow shape inference and canonicalization, so that specialized function
// always has ranked inputs and results to infer JitRt ABI requirements.
void CreateJitRtSpecializationPipeline(mlir::OpPassManager& pm);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_JITRT_PIPELINE_H_
