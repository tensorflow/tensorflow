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

#include "tensorflow/compiler/mlir/quantization/stablehlo/utils/fill_quantization_options.h"

#include <ostream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_options.pb.h"
#include "tensorflow/tsl/platform/protobuf.h"

namespace mlir::stablehlo {
namespace {

using ::stablehlo::quantization::PresetQuantizationMethod;
using ::stablehlo::quantization::QuantizationComponentSpec;
using ::stablehlo::quantization::QuantizationOptions;

// Simple implementation of ::testing::EqualsProto equivalent until open source
// b/135192747 is fixed. Originally from type_to_shape_test.cc.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tsl::protobuf::Message& expected)
      : expected_(expected.SerializeAsString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p, testing::MatchResultListener*) const {
    return p.SerializeAsString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tsl::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

void FillPresetQuantizationOptionsTestHelper(
    const PresetQuantizationMethod::PresetMethod preset_quantization_options,
    const QuantizationComponentSpec expected_activation_component,
    const QuantizationComponentSpec expected_weight_component,
    const QuantizationComponentSpec expected_bias_component) {
  QuantizationOptions quantization_options;
  quantization_options.mutable_quantization_method()
      ->mutable_preset_quantization_method()
      ->set_preset_method(preset_quantization_options);
  QuantizationOptions filled_quantization_options =
      FillPresetQuantizationOptions(quantization_options);
  for (QuantizationComponentSpec component :
       filled_quantization_options.quantization_method()
           .custom_quantization_method()
           .quantization_component_spec()) {
    switch (component.quantization_component()) {
      case (QuantizationComponentSpec::COMPONENT_ACTIVATION):
        EXPECT_THAT(component, EqualsProto(expected_activation_component));
        break;
      case (QuantizationComponentSpec::COMPONENT_WEIGHT):
        EXPECT_THAT(component, EqualsProto(expected_weight_component));
        break;
      case (QuantizationComponentSpec::COMPONENT_BIAS):
        EXPECT_THAT(component, EqualsProto(expected_bias_component));
        break;
      default:
        break;
    }
  }
}

TEST(FillQuantizationOptionsTest, PresetFloat16) {
  QuantizationComponentSpec activation_component, weight_component,
      bias_component;
  weight_component.set_quantization_component(
      QuantizationComponentSpec::COMPONENT_WEIGHT);
  weight_component.set_bit_width(QuantizationComponentSpec::BIT_WIDTH_16);
  weight_component.set_bit_type(QuantizationComponentSpec::BIT_TYPE_FLOAT);
  bias_component.set_quantization_component(
      QuantizationComponentSpec::COMPONENT_BIAS);
  bias_component.set_bit_width(QuantizationComponentSpec::BIT_WIDTH_16);
  bias_component.set_bit_type(QuantizationComponentSpec::BIT_TYPE_FLOAT);

  FillPresetQuantizationOptionsTestHelper(
      /*preset_quantization_options=*/PresetQuantizationMethod::FLOAT16,
      /*expected_activation_component=*/activation_component,
      /*expected_weight_component*/ weight_component,
      /*expected_bias_component*/ bias_component);
}

}  // namespace
}  // namespace mlir::stablehlo
