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

#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

using ::stablehlo::quantization::CustomQuantizationMethod;
using ::stablehlo::quantization::PresetQuantizationMethod;
using ::stablehlo::quantization::QuantizationComponentSpec;
using ::stablehlo::quantization::QuantizationOptions;
using QuantizationComponent =
    ::stablehlo::quantization::QuantizationComponentSpec_QuantizationComponent;
using BitType = ::stablehlo::quantization::QuantizationComponentSpec_BitType;
using BitWidth = ::stablehlo::quantization::QuantizationComponentSpec_BitWidth;

// Sets component, bit type and bit width information to the given spec
// instance.
void SetQuantizationComponentSpec(QuantizationComponentSpec* spec,
                                  const QuantizationComponent& component,
                                  const BitType bit_type,
                                  const BitWidth bit_width) {
  spec->set_quantization_component(component);
  spec->set_bit_type(bit_type);
  spec->set_bit_width(bit_width);
}

::stablehlo::quantization::QuantizationOptions FillPresetQuantizationOptions(
    ::stablehlo::quantization::QuantizationOptions quantization_options_) {
  CustomQuantizationMethod custom_method =
      quantization_options_.quantization_method().custom_quantization_method();
  QuantizationComponentSpec *activation_component, *weight_component,
      *bias_component;
  const auto preset_method = quantization_options_.quantization_method()
                                 .preset_quantization_method()
                                 .preset_method();
  if (!preset_method) return quantization_options_;
  switch (preset_method) {
    case PresetQuantizationMethod::FLOAT16:
      weight_component = custom_method.add_quantization_component_spec();
      SetQuantizationComponentSpec(weight_component,
                                   QuantizationComponentSpec::COMPONENT_WEIGHT,
                                   QuantizationComponentSpec::BIT_TYPE_FLOAT,
                                   QuantizationComponentSpec::BIT_WIDTH_16);
      bias_component = custom_method.add_quantization_component_spec();
      SetQuantizationComponentSpec(bias_component,
                                   QuantizationComponentSpec::COMPONENT_BIAS,
                                   QuantizationComponentSpec::BIT_TYPE_FLOAT,
                                   QuantizationComponentSpec::BIT_WIDTH_16);
      break;
    // Note: This is weight-only quantization by default, but with the legacy
    // flag "--force_dynamic_range_in_kernel", a DRQ behavior will be forced
    // in the kernel.
    case PresetQuantizationMethod::WEIGHT_ONLY:
      weight_component = custom_method.add_quantization_component_spec();
      SetQuantizationComponentSpec(weight_component,
                                   QuantizationComponentSpec::COMPONENT_WEIGHT,
                                   QuantizationComponentSpec::BIT_TYPE_INT,
                                   QuantizationComponentSpec::BIT_WIDTH_8);
      break;
    case PresetQuantizationMethod::POST_TRAINING_QUANTIZATION_STATIC_RANGE:
      activation_component = custom_method.add_quantization_component_spec();
      SetQuantizationComponentSpec(
          activation_component, QuantizationComponentSpec::COMPONENT_ACTIVATION,
          QuantizationComponentSpec::BIT_TYPE_INT,
          QuantizationComponentSpec::BIT_WIDTH_8);
      weight_component = custom_method.add_quantization_component_spec();
      SetQuantizationComponentSpec(weight_component,
                                   QuantizationComponentSpec::COMPONENT_WEIGHT,
                                   QuantizationComponentSpec::BIT_TYPE_INT,
                                   QuantizationComponentSpec::BIT_WIDTH_8);
      bias_component = custom_method.add_quantization_component_spec();
      SetQuantizationComponentSpec(bias_component,
                                   QuantizationComponentSpec::COMPONENT_BIAS,
                                   QuantizationComponentSpec::BIT_TYPE_INT,
                                   QuantizationComponentSpec::BIT_WIDTH_32);
      break;
    default:
      break;
  }
  *quantization_options_.mutable_quantization_method()
       ->mutable_custom_quantization_method() = custom_method;
  return quantization_options_;
}

LogicalResult GetActivationBitWidth(QuantizationOptions quantization_options,
                                    int* bit_width) {
  CustomQuantizationMethod custom_method =
      quantization_options.quantization_method().custom_quantization_method();

  // TODO: b/288046643 - Look up bit width for each op/op instance instead of
  // global configuration per component.
  for (const auto& component : custom_method.quantization_component_spec()) {
    if (component.quantization_component() ==
        QuantizationComponentSpec::COMPONENT_ACTIVATION) {
      switch (component.bit_width()) {
        case QuantizationComponentSpec::BIT_WIDTH_4:
          *bit_width = 4;
          return success();
          break;
        case QuantizationComponentSpec::BIT_WIDTH_8:
          *bit_width = 8;
          return success();
          break;
        case QuantizationComponentSpec::BIT_WIDTH_16:
          *bit_width = 16;
          return success();
          break;
        case QuantizationComponentSpec::BIT_WIDTH_32:
          *bit_width = 32;
          return success();
          break;
        default:
          break;
      }
    }
  }
  return failure();
}

}  // namespace mlir::quant::stablehlo
