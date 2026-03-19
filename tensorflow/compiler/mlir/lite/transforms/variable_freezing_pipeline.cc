/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/variable_freezing_pipeline.h"

#include "tensorflow/compiler/mlir/lite/transforms/pass_registry_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/unfreeze_global_constants.h"
#include "tensorflow/compiler/mlir/lite/transforms/variable_freezing_pipeline_options.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"

namespace mlir {
namespace TFL {

void VariableFreezingPipeline::AddPasses() {
  AddPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass(),
          [](const VariableFreezingPipelineOptions& options) { return true; });
  AddPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(true),
          [](const VariableFreezingPipelineOptions& options) { return true; });
  AddPass(mlir::TFL::Create<mlir::TFL::UnfreezeMutableGlobalTensorsPass>(),
          [](const VariableFreezingPipelineOptions& options) { return true; });
}

}  // namespace TFL
}  // namespace mlir
