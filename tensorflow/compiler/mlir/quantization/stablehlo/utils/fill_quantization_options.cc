/* Copyright 2023 The StableHLO Authors. All Rights Reserved.

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

#include "llvm/Support/Debug.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"

namespace mlir {
namespace stablehlo {

using ::stablehlo::quantization::CustomQuantizationMethod;
using ::stablehlo::quantization::PresetQuantizationMethod;
using ::stablehlo::quantization::QuantizationComponentSpec;

// Returns QuantizationOptions filled with detailed specs when user specifies
// an optional preset method name. The preset methods are defined in
// quantization_options.proto. This function will only be executed if a user
// gives a preset method, not a custom method.
::stablehlo::quantization::QuantizationOptions FillPresetQuantizationOptions(
    ::stablehlo::quantization::QuantizationOptions quantization_options_) {
  CustomQuantizationMethod custom_method =
      quantization_options_.quantization_method().custom_quantization_method();
  QuantizationComponentSpec *weight_component, *bias_component;
  auto preset_method = quantization_options_.quantization_method()
                           .preset_quantization_method()
                           .preset_method();
  if (!preset_method) return quantization_options_;
  switch (preset_method) {
    case PresetQuantizationMethod::FLOAT16:
      weight_component = custom_method.add_quantization_component_spec();
      weight_component->set_quantization_component(
          QuantizationComponentSpec::COMPONENT_WEIGHT);
      weight_component->set_bit_width(QuantizationComponentSpec::BIT_WIDTH_16);
      weight_component->set_bit_type(QuantizationComponentSpec::BIT_TYPE_FLOAT);
      bias_component = custom_method.add_quantization_component_spec();
      bias_component->set_quantization_component(
          QuantizationComponentSpec::COMPONENT_WEIGHT);
      bias_component->set_bit_width(QuantizationComponentSpec::BIT_WIDTH_16);
      bias_component->set_bit_type(QuantizationComponentSpec::BIT_TYPE_FLOAT);
      break;
    // Note: This is weight-only quantization by default, but with the legacy
    // flag "--force_dynamic_range_in_kernel", a DRQ behavior will be forced
    // in the kernel.
    case PresetQuantizationMethod::WEIGHT_ONLY:
      weight_component = custom_method.add_quantization_component_spec();
      weight_component->set_quantization_component(
          QuantizationComponentSpec::COMPONENT_WEIGHT);
      weight_component->set_bit_width(QuantizationComponentSpec::BIT_WIDTH_8);
      weight_component->set_bit_type(QuantizationComponentSpec::BIT_TYPE_INT);
      break;
    default:
      break;
  }
  *quantization_options_.mutable_quantization_method()
       ->mutable_custom_quantization_method() = custom_method;
  return quantization_options_;
}

}  // namespace stablehlo
}  // namespace mlir
