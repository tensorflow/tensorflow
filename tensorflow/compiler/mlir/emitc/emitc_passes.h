/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

#ifndef TENSORFLOW_COMPILER_MLIR_EMITC_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_EMITC_PASSES_H_

#include <optional>
#include <string>

#include "llvm/Support/CommandLine.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"      // from @llvm-project

namespace mlir {
namespace emitc {
struct AddReflectionMapPipelineOptions
    : public PassPipelineOptions<AddReflectionMapPipelineOptions> {
  AddReflectionMapPipelineOptions() {}
};

void createAddReflectionMapPipeline(
    mlir::OpPassManager& pm, const AddReflectionMapPipelineOptions& opts);

void registerAddReflectionMapPipeline();

}  // namespace emitc
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_EMITC_PASSES_H_
