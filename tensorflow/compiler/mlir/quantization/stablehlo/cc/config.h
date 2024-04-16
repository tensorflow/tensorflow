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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CONFIG_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CONFIG_H_

#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace stablehlo::quantization {

// Returns a copy of `user_provided_config` with default values populated where
// the user did not explicitly specify.
QuantizationConfig PopulateDefaults(
    const QuantizationConfig& user_provided_config);

// Returns a copy of `QuantizationConfig` where presets are expanded and
// transformed into other fields in `QuantizationConfig`.
//
// The expansion rules are as follows:
// * StaticRangePtqPreset
//   - The preset's `representative_datasets` field will be transferred to
//   `QuantizationConfig.calibration_options.representative_datasets`, unless
//   the user explicitly provided representative dataset configs to
//   `calibration_options`. In that case, the explicit configs take precedence
//   and the preset's configs are ignored.
//   - For `QuantizationSpecs`, the expanded `QuantizationSpec`s will be
//   populated first and user-provided `QuantizationSpec`s, if any, will be
//   appended. This expresses the fact that user-provided specs take precedence.
// * Preset unspecified
//   - No-op.
QuantizationConfig ExpandPresets(const QuantizationConfig& config);

// Returns whether a given QuantizationSpecs has the given quantization method.
bool HasQuantizationMethod(const QuantizationSpecs& specs,
                           Method::MethodCase method_case);

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CONFIG_H_
