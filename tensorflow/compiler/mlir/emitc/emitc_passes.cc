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

#include "tensorflow/compiler/mlir/emitc/emitc_passes.h"

#include "mlir/Dialect/EmitC/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"                 // from @llvm-project
#include "mlir/Pass/PassRegistry.h"                // from @llvm-project
#include "mlir/Transforms/Passes.h"                // from @llvm-project
#include "tensorflow/compiler/mlir/emitc/transforms/passes.h"

namespace mlir {
namespace emitc {
void createAddReflectionMapPipeline(
    mlir::OpPassManager& pm, const AddReflectionMapPipelineOptions& opts) {
  pm.addPass(mlir::emitc::CreateAddReflectionMapPass());
}

void registerAddReflectionMapPipeline() {
  mlir::PassPipelineRegistration<AddReflectionMapPipelineOptions>(
      "add-reflection-map-pipeline", "Add a reflection map to EmitC class",
      createAddReflectionMapPipeline);
}

}  // namespace emitc
}  // namespace mlir
