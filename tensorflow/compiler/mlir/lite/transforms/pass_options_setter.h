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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_OPTIONS_SETTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_OPTIONS_SETTER_H_

namespace mlir {
namespace TFL {

class OptimizePassOptions;
class VariableFreezingPipelineOptions;
class EmptyPassOptions;

// Interface for setting options for TFLite Converter Pass/Pipeline Options.
class PassOptionsSetter {
 public:
  virtual ~PassOptionsSetter() = default;
  virtual void SetOptions(OptimizePassOptions& options) const = 0;
  virtual void SetOptions(VariableFreezingPipelineOptions& options) const = 0;
  virtual void SetOptions(EmptyPassOptions& options) const = 0;
};
}  // namespace TFL
}  // namespace mlir

#endif  //  TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_OPTIONS_SETTER_H_
