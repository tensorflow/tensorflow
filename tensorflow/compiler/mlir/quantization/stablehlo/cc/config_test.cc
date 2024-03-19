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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"

namespace stablehlo::quantization {
namespace {

using ::testing::Eq;

TEST(PopulateDefaultsTest, PopulateDefaultsForEmptyConfig) {
  QuantizationConfig config{};

  const QuantizationConfig new_config = PopulateDefaults(config);
  EXPECT_TRUE(new_config.pipeline_config().unpack_quantized_types());
}

TEST(PopulateDefaultsTest, PopulateDefaultsForConfigWithUnpackQuantizedTypes) {
  QuantizationConfig config{};
  config.mutable_pipeline_config()->set_unpack_quantized_types(false);

  // Test that if the user explicitly provided `unpack_quantized_types`, it is
  // not overridden.
  const QuantizationConfig new_config = PopulateDefaults(config);
  EXPECT_FALSE(new_config.pipeline_config().unpack_quantized_types());
}

TEST(PopulateDefaultsTest, DefaultCalibrationOptionsPopulated) {
  QuantizationConfig config{};

  const QuantizationConfig new_config = PopulateDefaults(config);
  EXPECT_THAT(new_config.calibration_options().calibration_method(),
              Eq(CalibrationOptions::CALIBRATION_METHOD_MIN_MAX));
}

TEST(PopulateDefaultsTest, ExplicitCalibrationOptionsNotOverridden) {
  QuantizationConfig config{};
  CalibrationOptions& calibration_options =
      *config.mutable_calibration_options();
  calibration_options.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX);
  calibration_options.mutable_calibration_parameters()->set_initial_num_bins(
      512);

  // Test that if the user explicitly provided `calibration_options`, it is not
  // overridden.
  const QuantizationConfig new_config = PopulateDefaults(config);
  EXPECT_THAT(new_config.calibration_options().calibration_method(),
              Eq(CalibrationOptions::CALIBRATION_METHOD_AVERAGE_MIN_MAX));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .initial_num_bins(),
              Eq(512));
}

}  // namespace
}  // namespace stablehlo::quantization
