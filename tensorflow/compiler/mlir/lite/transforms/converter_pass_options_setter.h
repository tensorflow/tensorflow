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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CONVERTER_PASS_OPTIONS_SETTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CONVERTER_PASS_OPTIONS_SETTER_H_

#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/converter_flags.pb.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options_setter.h"

namespace mlir {
namespace TFL {

class OptimizePassOptions;
class VariableFreezingPipelineOptions;
class EmptyPassOptions;

// PassOptionsSetter to set TFLite Converter Pass/Pipeline Options based on
// ConverterFlags and TFL::PassConfig values.
class ConverterPassOptionsSetter : public PassOptionsSetter {
 public:
  explicit ConverterPassOptionsSetter(
      const tflite::ConverterFlags& converter_flags,
      const mlir::TFL::PassConfig& pass_config)
      : converter_flags_(converter_flags), pass_config_(pass_config) {};
  ~ConverterPassOptionsSetter() override = default;

  void SetOptions(OptimizePassOptions& options) const override;
  void SetOptions(VariableFreezingPipelineOptions& options) const override;
  void SetOptions(EmptyPassOptions& options) const override;

 private:
  tflite::ConverterFlags converter_flags_;
  mlir::TFL::PassConfig pass_config_;
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CONVERTER_PASS_OPTIONS_SETTER_H_
