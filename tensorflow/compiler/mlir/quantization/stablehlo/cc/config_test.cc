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
using ::testing::SizeIs;
using ::testing::StrEq;

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

TEST(PopulateDefaultsTest,
     DefaultCalibrationOptionsPopulatedForUnspecifiedMethod) {
  QuantizationConfig config{};
  CalibrationOptions& calibration_options =
      *config.mutable_calibration_options();
  calibration_options.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_UNSPECIFIED);

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

TEST(PopulateDefaultsTest, DefaultNumbersPopulatedForPartOfCalibrationOptions) {
  QuantizationConfig config{};
  CalibrationOptions& calibration_options =
      *config.mutable_calibration_options();
  calibration_options.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE);
  calibration_options.mutable_calibration_parameters()->set_initial_num_bins(
      512);

  // Test that if the user explicitly provided part of the
  // `calibration_options`, it is not overridden, rest of the data are default.
  const QuantizationConfig new_config = PopulateDefaults(config);
  EXPECT_THAT(new_config.calibration_options().calibration_method(),
              Eq(CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_PERCENTILE));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .initial_num_bins(),
              Eq(512));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .min_percentile(),
              Eq(0.001f));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .max_percentile(),
              Eq(99.999f));
}

TEST(PopulateDefaultsTest,
     DefaultNumbersPopulatedForCalibrationOptionsOfHistogramMseBruteforce) {
  QuantizationConfig config{};
  CalibrationOptions& calibration_options =
      *config.mutable_calibration_options();
  calibration_options.set_calibration_method(
      CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE);

  const QuantizationConfig new_config = PopulateDefaults(config);
  EXPECT_THAT(
      new_config.calibration_options().calibration_method(),
      Eq(CalibrationOptions::CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .initial_num_bins(),
              Eq(256));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .min_percentile(),
              Eq(0.0f));
  EXPECT_THAT(new_config.calibration_options()
                  .calibration_parameters()
                  .max_percentile(),
              Eq(0.0f));
}

TEST(ExpandPresetsTest, ExpandUnspecifiedPreset) {
  QuantizationConfig config{};
  const QuantizationConfig new_config = ExpandPresets(config);

  // Test that nothing has been changed.
  EXPECT_FALSE(new_config.has_specs());
  EXPECT_FALSE(new_config.has_calibration_options());
  EXPECT_FALSE(new_config.has_pipeline_config());
}

TEST(ExpandPresetsTest, ExpandStaticRangePtqEnableFullIntquantization) {
  QuantizationConfig config{};
  RepresentativeDatasetConfig& preset_dataset_config =
      *config.mutable_static_range_ptq_preset()->add_representative_datasets();
  config.mutable_static_range_ptq_preset()->set_enable_full_int_quantization(
      true);
  preset_dataset_config.mutable_tf_record()->set_path("/test/path");

  const QuantizationConfig new_config = ExpandPresets(config);
  ASSERT_THAT(new_config.specs().specs(), SizeIs(2));

  const QuantizationSpec& default_spec = new_config.specs().specs(0);
  EXPECT_THAT(default_spec.matcher().function_name().regex(), StrEq(".*"));
  EXPECT_TRUE(default_spec.method().has_static_range_ptq());

  // Test that the expansion for convolution ops is done.
  const QuantizationSpec& conv_spec = new_config.specs().specs(1);
  EXPECT_THAT(conv_spec.matcher().function_name().regex(),
              StrEq("composite_conv.*"));
  ASSERT_TRUE(conv_spec.method().has_static_range_ptq());

  const StaticRangePtq& srq_spec = conv_spec.method().static_range_ptq();
  ASSERT_THAT(srq_spec.input_quantized_types(), SizeIs(1));
  ASSERT_TRUE(srq_spec.input_quantized_types().contains(1));

  EXPECT_THAT(
      srq_spec.input_quantized_types().at(1).dimension_specs().dimension(),
      Eq(3));

  // Test that representative dataset config has been transferred to the
  // `CalibrationOptions`.
  ASSERT_THAT(new_config.calibration_options().representative_datasets(),
              SizeIs(1));
  EXPECT_THAT(new_config.calibration_options()
                  .representative_datasets(0)
                  .tf_record()
                  .path(),
              StrEq("/test/path"));
}

TEST(ExpandPresetsTest, ExpandStaticRangePtqPresetDefault) {
  QuantizationConfig config{};
  RepresentativeDatasetConfig& preset_dataset_config =
      *config.mutable_static_range_ptq_preset()->add_representative_datasets();
  preset_dataset_config.mutable_tf_record()->set_path("/test/path");

  const QuantizationConfig new_config = ExpandPresets(config);
  ASSERT_THAT(new_config.specs().specs(), SizeIs(2));

  const QuantizationSpec& spec = new_config.specs().specs(0);
  EXPECT_THAT(spec.matcher().function_name().regex(),
              StrEq("^.*(dot_general|gather).*"));
  EXPECT_TRUE(spec.method().has_static_range_ptq());
}

TEST(ExpandPresetsTest,
     ExpandStaticRangePtqPresetWithTopLevelRepresentativeDataset) {
  // Test the scenario where both
  // `config.calibration_options.representative_datasets` and
  // `config.static_range_ptq_preset.representative_datasets` are both
  // specified. In this case, the one set to the `calibration_options` takes
  // precedence.
  QuantizationConfig config{};
  RepresentativeDatasetConfig& top_level_dataset_config =
      *config.mutable_calibration_options()->add_representative_datasets();
  top_level_dataset_config.mutable_tf_record()->set_path("/test/path/1");

  RepresentativeDatasetConfig& preset_dataset_config =
      *config.mutable_static_range_ptq_preset()->add_representative_datasets();
  preset_dataset_config.mutable_tf_record()->set_path("/test/path/2");

  const QuantizationConfig new_config = ExpandPresets(config);

  // Test that representative dataset config has not been transferred to the
  // `CalibrationOptions`. Top-level config takes precedence.
  ASSERT_THAT(new_config.calibration_options().representative_datasets(),
              SizeIs(1));
  EXPECT_THAT(new_config.calibration_options()
                  .representative_datasets(0)
                  .tf_record()
                  .path(),
              StrEq("/test/path/1"));
}

TEST(ExpandPresetsTest, ExpandStaticRangePtqPresetThenAppendExplicitSpecs) {
  QuantizationConfig config{};
  config.mutable_static_range_ptq_preset()->set_enable_full_int_quantization(
      true);

  QuantizationSpec& user_provided_spec = *config.mutable_specs()->add_specs();
  user_provided_spec.mutable_matcher()->mutable_function_name()->set_regex(
      "composite_dot_general_fn_1");
  user_provided_spec.mutable_method()->mutable_no_quantization();

  // Test that the expanded `QuantizationSpec`s are populated first and then
  // user-provided specs are appended.
  //
  // It should look like:
  //
  // specs {matcher {function_name {regex: ".*"}} method {static_range_ptq {}}}
  // specs {
  //   matcher {function_name {regex: "composite_conv.*"}}
  //   method {static_range_ptq {...}}}
  // }
  // specs {
  //   matcher {function_name {regex: "composite_dot_general_fn_1"}}
  //   method {no_quantization {}}
  // }
  const QuantizationConfig new_config = ExpandPresets(config);
  ASSERT_THAT(new_config.specs().specs(), SizeIs(3));

  const QuantizationSpec& first_spec = new_config.specs().specs(0);
  EXPECT_THAT(first_spec.matcher().function_name().regex(), StrEq(".*"));
  EXPECT_TRUE(first_spec.method().has_static_range_ptq());

  const QuantizationSpec& second_spec = new_config.specs().specs(1);
  EXPECT_THAT(second_spec.matcher().function_name().regex(),
              StrEq("composite_conv.*"));
  EXPECT_TRUE(second_spec.method().has_static_range_ptq());

  // This corresponds to `user_provided_spec`.
  const QuantizationSpec& third_spec = new_config.specs().specs(2);
  EXPECT_THAT(third_spec.matcher().function_name().regex(),
              StrEq("composite_dot_general_fn_1"));
  EXPECT_TRUE(third_spec.method().has_no_quantization());
}

TEST(ExpandPresetsTest, ExpandWeightOnlyPtqPresetDefault) {
  QuantizationConfig config{};
  *config.mutable_weight_only_ptq_preset() = WeightOnlyPtqPreset();

  const QuantizationConfig new_config = ExpandPresets(config);
  ASSERT_THAT(new_config.specs().specs(), SizeIs(1));

  const QuantizationSpec& spec = new_config.specs().specs(0);
  EXPECT_THAT(spec.matcher().function_name().regex(),
              StrEq("^.*(conv|dot_general).*"));
  EXPECT_TRUE(spec.method().has_weight_only_ptq());
}

}  // namespace
}  // namespace stablehlo::quantization
