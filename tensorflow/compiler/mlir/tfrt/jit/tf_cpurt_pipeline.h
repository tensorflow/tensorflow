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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_PIPELINE_H_

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "llvm/ADT/Hashing.h"

namespace tensorflow {

struct TfCpuRtPipelineOptions
    : public mlir::PassPipelineOptions<TfCpuRtPipelineOptions> {
  Option<bool> vectorize{*this, "vectorize",
                         llvm::cl::desc("Enable tiling for vectorization."),
                         llvm::cl::init(false)};
};

// Make TfCpuRtPipelineOptions hashable.
inline ::llvm::hash_code hash_value(const TfCpuRtPipelineOptions& opts) {
  return ::llvm::hash_value(static_cast<bool>(opts.vectorize));
}

// Creates a pipeline that lowers modules from the Tensorflow dialect to
// the Linalg on buffers. `TfCpuRtPipelineOptions` contains flags to
// enable/disable experimental features.
void CreateTfCpuRtPipeline(mlir::OpPassManager& pm,
                           const TfCpuRtPipelineOptions& options);

// Calls CreateTfCpuRtPipeline with the default TfCpuRtPipelineOptions.
void CreateDefaultTfCpuRtPipeline(mlir::OpPassManager& pm);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_JIT_TF_CPURT_PIPELINE_H_
