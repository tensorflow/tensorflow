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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"

namespace stablehlo::quantization {

QuantizationConfig PopulateDefaults(
    const QuantizationConfig& user_provided_config) {
  QuantizationConfig config = user_provided_config;

  PipelineConfig& pipeline_config = *config.mutable_pipeline_config();
  if (!pipeline_config.has_unpack_quantized_types()) {
    pipeline_config.set_unpack_quantized_types(true);
  }

  return config;
}

}  // namespace stablehlo::quantization
