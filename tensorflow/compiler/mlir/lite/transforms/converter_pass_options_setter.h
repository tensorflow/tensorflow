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
class OptimizeBroadcastLikePassOptions;

// PassOptionsSetter to set TFLite Converter Pass/Pipeline Options based on
// ConverterFlags and TFL::PassConfig values.
//
// INVARIANT: the SetOptions overloads declared below MUST match the
// definitions in converter_pass_options_setter.cc EXACTLY (same parameter
// types, same constness, same override). The Windows link invokes
// `lld-link`, which fails hard if any caller references an overload
// declared here that lacks a corresponding definition. If you add an
// overload, add it in both places.
//
// If the Windows build fails with "undefined symbol:
// ?SetOptions@ConverterPassOptionsSetter@..." pointing at an overload
// that exists in this header, the most likely cause is a stale Bazel
// cache: run `bazel clean --expunge` on the Windows runner before
// suspecting source. The next-most-likely cause is that tf_tfl_passes.cc
// was built from a different source tree than this header.
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
  void SetOptions(OptimizeBroadcastLikePassOptions& options) const override;

 private:
  tflite::ConverterFlags converter_flags_;
  mlir::TFL::PassConfig pass_config_;
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_CONVERTER_PASS_OPTIONS_SETTER_H_
