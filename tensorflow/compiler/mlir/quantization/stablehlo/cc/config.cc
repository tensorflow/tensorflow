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

#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace stablehlo::quantization {
namespace {

// Creates `CalibrationOptions` with default fields. Uses simple min-max
// calibration by default.
CalibrationOptions GetDefaultCalibrationOptions() {
  CalibrationOptions options{};
  options.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);

  return options;
}

void ExpandStaticRangePtqPreset(const StaticRangePtqPreset& preset,
                                QuantizationConfig& config) {
  // Populate with preset's representative dataset configs if the user didn't
  // explicitly specify other representative dataset configs to the top-level
  // `CalibrationOptions`.
  if (config.calibration_options().representative_datasets().empty()) {
    auto preset_datasets = preset.representative_datasets();
    config.mutable_calibration_options()
        ->mutable_representative_datasets()
        ->Add(preset_datasets.begin(), preset_datasets.end());
  }

  // Create a new `QuantizationSpecs` to replace the existing one. The expansion
  // from `StaticRangePtqPreset` gets populated first and then user-provided
  // explicit `QuantizationSpec`s will be appended.
  QuantizationSpecs new_specs{};
  QuantizationSpec& spec = *new_specs.add_specs();
  spec.mutable_matcher()->mutable_function_name()->set_regex(".*");
  spec.mutable_method()->mutable_static_range_ptq();

  const QuantizationSpecs& previous_specs = config.specs();
  new_specs.mutable_specs()->Add(previous_specs.specs().begin(),
                                 previous_specs.specs().end());

  config.mutable_specs()->Swap(&new_specs);
}

}  // namespace

QuantizationConfig ExpandPresets(const QuantizationConfig& config) {
  QuantizationConfig new_config = config;

  // Update the `new_config` with each preset's expansions.
  switch (config.preset_case()) {
    case QuantizationConfig::kStaticRangePtqPreset:
      ExpandStaticRangePtqPreset(config.static_range_ptq_preset(), new_config);
      break;
    default:
      // Preset has not been specified. The expansion is a no-op.
      break;
  }

  return new_config;
}

QuantizationConfig PopulateDefaults(
    const QuantizationConfig& user_provided_config) {
  QuantizationConfig config = user_provided_config;

  if (!config.has_calibration_options()) {
    *config.mutable_calibration_options() = GetDefaultCalibrationOptions();
  }

  PipelineConfig& pipeline_config = *config.mutable_pipeline_config();
  if (!pipeline_config.has_unpack_quantized_types()) {
    pipeline_config.set_unpack_quantized_types(true);
  }

  return config;
}

}  // namespace stablehlo::quantization
