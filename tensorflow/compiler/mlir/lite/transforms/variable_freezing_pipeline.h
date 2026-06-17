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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_VARIABLE_FREEZING_PIPELINE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_VARIABLE_FREEZING_PIPELINE_H_

#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/pipeline.h"
#include "tensorflow/compiler/mlir/lite/transforms/variable_freezing_pipeline_options.h"

namespace mlir {
namespace TFL {

class VariableFreezingPipeline
    : public Pipeline<VariableFreezingPipeline,
                      VariableFreezingPipelineOptions> {
 public:
  void AddPasses() override;

  /// Returns the command-line argument attached to this pass.
  static llvm::StringRef GetArgument() {
    return "tfl-variable-freezing-pipeline";
  }

  static llvm::StringRef GetDescription() {
    return "Variable Freezing Pipeline";
  }

  /// Returns the derived pass name.
  static llvm::StringRef GetName() { return "VariableFreezingPipeline"; }
};

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_VARIABLE_FREEZING_PIPELINE_H_
