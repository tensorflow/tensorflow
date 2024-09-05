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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TOCO_PASS_OPTIONS_SETTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TOCO_PASS_OPTIONS_SETTER_H_

#include "tensorflow/compiler/mlir/lite/common/tfl_pass_config.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options_setter.h"
#include "tensorflow/lite/toco/toco_flags.pb.h"

namespace mlir {
namespace TFL {

class OptimizePass;
class EmptyPassOptions;

// PassOptionsSetter to set TFLite Converter Pass/Pipeline Options based on
// TocoFlags and TFL::PassConfig values.
class TocoPassOptionsSetter : public PassOptionsSetter {
 public:
  explicit TocoPassOptionsSetter(const toco::TocoFlags& toco_flags,
                                 const mlir::TFL::PassConfig& pass_config)
      : toco_flags_(toco_flags), pass_config_(pass_config) {};
  ~TocoPassOptionsSetter() override = default;

  void SetOptions(OptimizePassOptions& options) const override;
  void SetOptions(EmptyPassOptions& options) const override;

 private:
  toco::TocoFlags toco_flags_;
  mlir::TFL::PassConfig pass_config_;
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TOCO_PASS_OPTIONS_SETTER_H_
