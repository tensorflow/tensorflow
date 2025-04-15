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

#ifndef TENSORFLOW_COMPILER_MLIR_TOSA_TFL_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_TOSA_TFL_PASSES_H_

#include <optional>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace tosa {

struct TOSATFLLegalizationPipelineOptions
    : public PassPipelineOptions<TOSATFLLegalizationPipelineOptions> {
  ArrayRef<std::string> disabled_patterns;
  ArrayRef<std::string> enabled_patterns;

  PassOptions::Option<bool> target_compilation_backend{
      *this, "target-compilation-backend",
      llvm::cl::desc("Whether targetting compilation backend"),
      llvm::cl::init(false)};

  PassOptions::Option<bool> dequantize_tfl_softmax{
      *this, "dequantize-tfl-softmax",
      llvm::cl::desc("Dequantize the TFLite softmax"), llvm::cl::init(false)};

  TOSATFLLegalizationPipelineOptions() {
    disabled_patterns = std::nullopt;
    enabled_patterns = std::nullopt;
  }
};

// Legalizes TFL (TensorFlow lite) dialect(s) to Tosa.
void createTFLtoTOSALegalizationPipeline(
    OpPassManager& pm, const TOSATFLLegalizationPipelineOptions& opts);

void registerTFLtoTOSALegalizationPipeline();

}  // namespace tosa
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TOSA_TFL_PASSES_H_
