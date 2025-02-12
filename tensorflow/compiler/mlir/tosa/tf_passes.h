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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TF_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TOSA_TF_PASSES_H_

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project

namespace mlir::tosa {

struct TOSATFLegalizationPipelineOptions
    : public PassPipelineOptions<TOSATFLegalizationPipelineOptions> {};

// Legalizes TF dialect(s) to Tosa.
void createTFtoTOSALegalizationPipeline(
    OpPassManager& pm, const TOSATFLegalizationPipelineOptions& opts);

void registerTFtoTOSALegalizationPipeline();

}  // namespace mlir::mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TF_PASSES_H_
