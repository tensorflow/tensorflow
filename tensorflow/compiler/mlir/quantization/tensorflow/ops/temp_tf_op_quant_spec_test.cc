/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/temp_tf_op_quant_spec.h"

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir::tf_quant {
namespace {

using QuantizationOptions = tensorflow::quantization::QuantizationOptions;
using QuantizationComponentSpec =
    tensorflow::quantization::QuantizationComponentSpec;

TEST(TfOpQuantSpecTest, WeightComponentSpecExist) {
  QuantizationOptions quant_options;
  QuantizationComponentSpec quant_spec;
  quant_spec.set_quantization_component(
      QuantizationComponentSpec::COMPONENT_WEIGHT);
  quant_spec.set_tensor_type(QuantizationComponentSpec::TENSORTYPE_INT_8);
  auto mutable_quant_method = quant_options.mutable_quantization_method();
  *mutable_quant_method->add_quantization_component_specs() = quant_spec;
  auto output = GetWeightComponentSpec(quant_options);
  EXPECT_TRUE(output.has_value());
}

TEST(TfOpQuantSpecTest, WeightComponentSpecDoNotExist) {
  QuantizationOptions quant_options;
  auto output = GetWeightComponentSpec(quant_options);
  EXPECT_FALSE(output.has_value());
}

}  // namespace
}  // namespace mlir::tf_quant
