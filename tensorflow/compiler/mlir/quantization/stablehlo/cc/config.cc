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

#include <utility>

#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace stablehlo::quantization {
namespace {

// Populate `CalibrationOptions` with default fields.
void PopulateDefaultCalibrationOptions(QuantizationConfig& quant_config) {
  if (!quant_config.has_calibration_options() ||
      quant_config.calibration_options().calibration_method() ==
          CalibrationOptions::CALIBRATION_METHOD_UNSPECIFIED) {
    quant_config.mutable_calibration_options()->set_calibration_method(
        CalibrationOptions::CALIBRATION_METHOD_MIN_MAX);
  }

  switch (quant_config.calibration_options().calibration_method()) {
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE:
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE:
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY:
    case CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC:
      if (quant_config.calibration_options()
              .calibration_parameters()
              .num_bins() == 0) {
        quant_config.mutable_calibration_options()
            ->mutable_calibration_parameters()
            ->set_num_bins(512);
      }
      if (quant_config.calibration_options().calibration_method() ==
          CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE) {
        if (quant_config.calibration_options()
                .calibration_parameters()
                .min_percentile() == 0) {
          quant_config.mutable_calibration_options()
              ->mutable_calibration_parameters()
              ->set_min_percentile(0.001);
        }
        if (quant_config.calibration_options()
                .calibration_parameters()
                .max_percentile() == 0) {
          quant_config.mutable_calibration_options()
              ->mutable_calibration_parameters()
              ->set_max_percentile(99.999);
        }
      }
      break;
    default:
      break;
  }
}

// Returns a default `QuantizationSpec` for performing static-range PTQ on all
// ops.
//
// In textproto, the spec corresponds to:
//
// {
//   {matcher {function_name {regex: ".*"}}
//   {method {static_range_ptq {}}}
// }
QuantizationSpec GetDefaultStaticRangePtqSpec(StaticRangePtqPreset preset) {
  QuantizationSpec spec{};
  // Default for all ops.
  spec.mutable_matcher()->mutable_function_name()->set_regex(
      preset.enable_full_int_quantization() ? ".*"
                                            : "^.*(dot_general|gather).*");
  spec.mutable_method()->mutable_static_range_ptq();

  return spec;
}

QuantizationSpec GetDefaultWeightOnlyPtqSpec() {
  QuantizationSpec spec{};
  spec.mutable_matcher()->mutable_function_name()->set_regex(
      "^.*(conv|dot_general).*");

  WeightOnlyPtq& weight_only_ptq_spec =
      *spec.mutable_method()->mutable_weight_only_ptq();
  if (auto [iter, inserted] =
          weight_only_ptq_spec.mutable_input_quantized_types()->try_emplace(1);
      inserted) {
    iter->second.mutable_dimension_specs();
  }
  return spec;
}

// Returns a `QuantizationSpec` for performing static-range PTQ on the
// convolution quantizable unit family. Enables per-channel quantization for
// weights, on the channel dimension.
//
// In textproto, the spec corresponds to:
//
// {
//   {matcher {function_name {regex: "composite_conv.*"}}}
//   {method {static_range_ptq
//     {input_quantized_types {
//       key: 1,
//       value {dimension_specs {dimension: 3}}}}
//   }}
// }
QuantizationSpec GetPtqSpecForConvolution(Method::MethodCase method_case) {
  QuantizationSpec spec{};
  if (method_case != Method::kStaticRangePtq) {
    return spec;
  }

  // Matches all convolution quantizable unit family.
  spec.mutable_matcher()->mutable_function_name()->set_regex(
      "composite_conv.*");

  // Enable per-channel quantization for convolution weights.
  QuantizedType conv_weight_quantized_type{};

  // Assumes NHWC format, specifying the channel dimension (3) as the
  // quantized axis.
  conv_weight_quantized_type.mutable_dimension_specs()->set_dimension(3);

  // The index of weight operands passed to lifted functions for convolution
  // is 1.
  StaticRangePtq& static_range_ptq_spec =
      *spec.mutable_method()->mutable_static_range_ptq();
  static_range_ptq_spec.mutable_input_quantized_types()->try_emplace(
      1, std::move(conv_weight_quantized_type));
  return spec;
};

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

  // Create a new `QuantizationSpecs` to replace the existing one. The
  // expansion from `StaticRangePtqPreset` gets populated first and then
  // user-provided explicit `QuantizationSpec`s will be appended.
  QuantizationSpecs new_specs{};
  *new_specs.add_specs() =
      GetDefaultStaticRangePtqSpec(/*preset=*/config.static_range_ptq_preset());
  *new_specs.add_specs() =
      GetPtqSpecForConvolution(Method::MethodCase::kStaticRangePtq);

  // Append user-provided specs to override existing specs.
  const QuantizationSpecs& previous_specs = config.specs();
  new_specs.mutable_specs()->Add(previous_specs.specs().begin(),
                                 previous_specs.specs().end());

  config.clear_static_range_ptq_preset();
  config.mutable_specs()->Swap(&new_specs);
}

void ExpandWeightOnlyPtqPreset(QuantizationConfig& config) {
  // Create a new `QuantizationSpecs` to replace the existing one. The
  // expansion from `WeightOnlyPtqPreset` gets populated first and then
  // user-provided explicit `QuantizationSpec`s will be appended.
  QuantizationSpecs new_specs{};
  *new_specs.add_specs() = GetDefaultWeightOnlyPtqSpec();
  // TODO: b/307625297 - Add per-channel weight only support.

  // Append user-provided specs to override existing specs.
  const QuantizationSpecs& previous_specs = config.specs();
  new_specs.mutable_specs()->Add(previous_specs.specs().begin(),
                                 previous_specs.specs().end());

  config.clear_weight_only_ptq_preset();
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
    case QuantizationConfig::kWeightOnlyPtqPreset:
      ExpandWeightOnlyPtqPreset(new_config);
      break;
    default:
      // Preset has not been specified. The expansion is a no-op.
      break;
  }

  return new_config;
}

bool HasQuantizationMethod(const QuantizationSpecs& specs,
                           Method::MethodCase method_case) {
  for (const auto& spec : specs.specs()) {
    if (spec.method().method_case() == method_case) {
      return true;
    }
  }
  return false;
}

QuantizationConfig PopulateDefaults(
    const QuantizationConfig& user_provided_config) {
  QuantizationConfig config = user_provided_config;

  PopulateDefaultCalibrationOptions(config);

  PipelineConfig& pipeline_config = *config.mutable_pipeline_config();
  if (!pipeline_config.has_unpack_quantized_types()) {
    pipeline_config.set_unpack_quantized_types(true);
  }

  return config;
}

}  // namespace stablehlo::quantization
