/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/quantization_utils.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/testing/util.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace utils {
namespace {

std::unique_ptr<FlatBufferModel> ReadModel(const char* model) {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir, model);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModel> ReadConvModel() {
  return ReadModel(internal::kConvModelWith0Plus10Weights);
}

using ::testing::ElementsAreArray;

class QuantizationUtilsTest : public testing::Test {
 protected:
  tflite::TestErrorReporter error_reporter_;
};

TEST_F(QuantizationUtilsTest, NumElements) {
  TensorT tensor;
  tensor.shape = {1, 2, 3, 4};
  uint64_t num_elements;
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 1 * 2 * 3 * 4);

  tensor.shape = {5};
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 5);

  tensor.shape = {};
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  // Scalars with empty shape have 1 element.
  EXPECT_EQ(num_elements, 1);

  tensor.shape = {1, 2, 3, -1};
  EXPECT_EQ(kTfLiteError, NumElements(tensor, &num_elements));
}

TEST_F(QuantizationUtilsTest, GetAsymmetricQuantizationParamsUnitRange) {
  const float float_min = -128.0;
  const float float_max = 127.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);

  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_EQ(zero_point, 0);
  EXPECT_NEAR(scale, 1, eps);
}

TEST_F(QuantizationUtilsTest,
       AsymmetricQuantizationParamsWithAllPositiveRange) {
  // The min should get nudged to include 0, so the effective range is [0, 6].
  const float float_min = 1.0;
  const float float_max = 6.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], 0.0);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_EQ(zero_point, -128);
  EXPECT_NEAR(scale, 6 / 255.0f, eps);
}

TEST_F(QuantizationUtilsTest,
       AsymmetricQuantizationParamsWithAllNegativeRange) {
  // The max should get nudged to include 0, so the effective range is [-6, 0].
  const float float_min = -6.0;
  const float float_max = -1.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], 0.0);
  EXPECT_EQ(params.min[0], float_min);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_EQ(zero_point, 127);
  EXPECT_NEAR(scale, 6 / 255.0f, eps);
}

TEST_F(QuantizationUtilsTest, AsymmetricQuantizationParamsWithZeroInRange) {
  const float float_min = -5.0;
  const float float_max = 1.0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_NEAR(scale, 6 / 255.0f, eps);
  EXPECT_GT(zero_point, quant_min);
  EXPECT_LT(zero_point, quant_max);
}

TEST_F(QuantizationUtilsTest, AsymmetricQuantizationParamsWithZeroMinMax) {
  const float float_min = 0;
  const float float_max = 0;
  const int quant_min = -128;
  const int quant_max = 127;
  QuantizationParametersT params;
  GetAsymmetricQuantizationParams(float_min, float_max, quant_min, quant_max,
                                  &params);
  ASSERT_EQ(params.max.size(), 1);
  ASSERT_EQ(params.min.size(), 1);
  ASSERT_EQ(params.scale.size(), 1);
  ASSERT_EQ(params.zero_point.size(), 1);
  EXPECT_EQ(params.max[0], float_max);
  EXPECT_EQ(params.min[0], float_min);
  int64_t zero_point = params.zero_point[0];
  float scale = params.scale[0];
  const float eps = 1e-7f;
  EXPECT_NEAR(scale, 0, eps);
  EXPECT_NEAR(zero_point, quant_min, eps);
  EXPECT_LT(zero_point, quant_max);
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantizationWithNullQParams) {
  // Set up an input with [3, 2, 2, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Channel 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Channel 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Channel 3.
  };
  const int channel_index = 0;

  // Create holder for output scale and data.
  std::vector<float> output_scales(3);
  std::vector<int8_t> output_data(3 * 2 * 2 * 2);

  // Call SymmetricPerChannelQuantization with quant_params as a null pointer
  // and verify the result.
  TensorT tensor = TensorT();
  tensor.quantization = nullptr;
  tensor.shape = {3, 2, 2, 2};
  SymmetricPerChannelQuantization(&tensor, input.data(), channel_index,
                                  &output_scales, &output_data,
                                  &error_reporter_);
  const std::vector<float> expected_output_scales = {0.0393700786, 0.0629921257,
                                                     0.0472440943};
  const std::vector<int8_t> expected_output_data = {
      76, 51, 127, -51, 76,  51,  127,  -51,   // Channel 1.
      16, 32, 48,  64,  79,  95,  111,  127,   // Channel 2.
      21, 0,  -21, -42, -64, -85, -106, -127,  // Channel 3.
  };
  EXPECT_THAT(output_scales, ElementsAreArray(expected_output_scales));
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantization) {
  // Set up an input with [3, 2, 2, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Channel 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Channel 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Channel 3.
  };
  const int32_t channel_index = 0;

  // Create holder for output scale and data.
  std::vector<float> output_scales(3);
  std::vector<int8_t> output_data(3 * 2 * 2 * 2);

  // Initialize pointer to quantization parameters
  TensorT tensor = TensorT();
  tensor.quantization = std::make_unique<QuantizationParametersT>();
  tensor.shape = {3, 2, 2, 2};
  FillPerChannelMinMax(input.data(), tensor.shape, channel_index,
                       tensor.quantization.get(), &error_reporter_);

  // Test that FillPerChanneMinMax worked
  const std::vector<float> expected_mins = {-2.0, 1.0, -6.0};
  const std::vector<float> expected_maxs = {5.0, 8.0, 1.0};
  EXPECT_THAT(tensor.quantization->min, ElementsAreArray(expected_mins));
  EXPECT_THAT(tensor.quantization->max, ElementsAreArray(expected_maxs));

  // Call SymmetricPerChannelQuantization with quant_params as a null pointer
  // and verify the result.
  SymmetricPerChannelQuantization(&tensor, input.data(), channel_index,
                                  &output_scales, &output_data,
                                  &error_reporter_);
  const std::vector<float> expected_output_scales = {0.0393700786, 0.0629921257,
                                                     0.0472440943};
  const std::vector<int8_t> expected_output_data = {
      76, 51, 127, -51, 76,  51,  127,  -51,   // Channel 1.
      16, 32, 48,  64,  79,  95,  111,  127,   // Channel 2.
      21, 0,  -21, -42, -64, -85, -106, -127,  // Channel 3.
  };
  EXPECT_THAT(output_scales, ElementsAreArray(expected_output_scales));
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantization2DTensor) {
  // Set up an input with [3, 8] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Batch 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Batch 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Batch 3.
  };
  const int32_t channel_index = 1;

  // Create holder for output scale and data.
  std::vector<float> output_scales(8);
  std::vector<int8_t> output_data(3 * 8);

  // Initialize pointer to quantization parameters
  TensorT tensor = TensorT();
  tensor.quantization = std::make_unique<QuantizationParametersT>();
  tensor.shape = {3, 8};
  FillPerChannelMinMax(input.data(), tensor.shape, channel_index,
                       tensor.quantization.get(), &error_reporter_);

  // Test that FillPerChanneMinMax worked
  const std::vector<float> expected_mins = {1.0,  0.0,  -1.0, -2.0,
                                            -3.0, -4.0, -5.0, -6.0};
  const std::vector<float> expected_maxs = {3.0, 2.0, 5.0, 4.0,
                                            5.0, 6.0, 7.0, 8.0};
  EXPECT_THAT(tensor.quantization->min, ElementsAreArray(expected_mins));
  EXPECT_THAT(tensor.quantization->max, ElementsAreArray(expected_maxs));

  // Call SymmetricPerChannelQuantization with quant_params as a null pointer
  // and verify the result.
  SymmetricPerChannelQuantization(&tensor, input.data(), channel_index,
                                  &output_scales, &output_data,
                                  &error_reporter_);
  const std::vector<float> expected_output_scales = {
      0.02362204724, 0.01574803149, 0.03937007874, 0.03149606299,
      0.03937007874, 0.04724409448, 0.05511811023, 0.06299212598};
  const std::vector<int8_t> expected_output_data = {
      127, 127, 127, -64, 76,  42,  91,  -32,  // Batch 1.
      42,  127, 76,  127, 127, 127, 127, 127,  // Batch 2.
      42,  0,   -25, -64, -76, -85, -91, -95,  // Batch 3.
  };
  EXPECT_THAT(output_scales, ElementsAreArray(expected_output_scales));
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelQuantizeValues) {
  // Set up an input with [3, 1, 1, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      13.0, 21.0,  // Channel 1.
      21.0, 22.0,  // Channel 2.
      31.0, 40.0,  // Channel 3.
  };
  const std::vector<float> scales_inv = {2, 0.5, 3};
  const std::vector<int32_t> dimension = {3, 1, 1, 2};
  const int channel_index = 0;

  // Create holder for output data.
  std::vector<int8_t> output_data(3 * 1 * 1 * 2);

  // Call SymmetricPerChannelQuantizeValues and verify the result.
  SymmetricPerChannelQuantizeValues(input.data(), scales_inv, dimension,
                                    channel_index, &output_data);
  const std::vector<int8_t> expected_output_data = {
      26, 42,   // Channel 1.
      11, 11,   // Channel 2.
      93, 120,  // Channel 3.
  };
  EXPECT_THAT(output_data, ElementsAreArray(expected_output_data));
}

TEST_F(QuantizationUtilsTest, FillPerChannelMinMax) {
  // Set up an input with [3, 1, 1, 2] size.
  const std::vector<float> input = {
      13.0, 21.0,  // Channel 1.
      21.0, 22.0,  // Channel 2.
      31.0, 40.0,  // Channel 3.
  };

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();
  std::vector<int> dimension = {3, 1, 1, 2};
  int32_t channel_dim_idx = 0;
  const std::vector<float> expected_mins = {13.0, 21.0, 31.0};
  const std::vector<float> expected_maxs = {21.0, 22.0, 40.0};

  FillPerChannelMinMax(input.data(), dimension, channel_dim_idx,
                       &quantization_params, &error_reporter_);

  EXPECT_EQ(quantization_params.min, expected_mins);
  EXPECT_EQ(quantization_params.max, expected_maxs);
  EXPECT_EQ(quantization_params.quantized_dimension, channel_dim_idx);
}

TEST_F(QuantizationUtilsTest, FillPerChannelMinMaxFillDim3) {
  // Set up an input with [3, 1, 1, 2] size.
  const std::vector<float> input = {
      // Channel 1, Channel 2
      13.0, 21.0, 21.0, 22.0, 31.0, 40.0,
  };

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();
  std::vector<int> dimension = {3, 1, 1, 2};
  int32_t channel_dim_idx = 3;
  const std::vector<float> expected_mins = {13.0, 21.0};
  const std::vector<float> expected_maxs = {31.0, 40.0};

  FillPerChannelMinMax(input.data(), dimension, channel_dim_idx,
                       &quantization_params, &error_reporter_);

  EXPECT_EQ(quantization_params.min, expected_mins);
  EXPECT_EQ(quantization_params.max, expected_maxs);
  EXPECT_EQ(quantization_params.quantized_dimension, channel_dim_idx);
}

TEST_F(QuantizationUtilsTest, FillPerChannelMinMax2DTensor) {
  // Set up an input with [3, 2] size.
  const std::vector<float> input = {
      // Channel 1, Channel 2
      13.0, 21.0, 21.0, 22.0, 31.0, 40.0,
  };

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();
  std::vector<int> dimension = {3, 2};
  int32_t channel_dim_idx = 1;
  const std::vector<float> expected_mins = {13.0, 21.0};
  const std::vector<float> expected_maxs = {31.0, 40.0};

  FillPerChannelMinMax(input.data(), dimension, channel_dim_idx,
                       &quantization_params, &error_reporter_);

  EXPECT_EQ(quantization_params.min, expected_mins);
  EXPECT_EQ(quantization_params.max, expected_maxs);
  EXPECT_EQ(quantization_params.quantized_dimension, channel_dim_idx);
}

TEST_F(QuantizationUtilsTest, FillSingleMinMax) {
  // Set up an input with [3, 1, 1, 2] size.
  const std::vector<float> input = {
      13.0, 21.0,  // Channel 1.
      21.0, 22.0,  // Channel 2.
      31.0, 40.0,  // Channel 3.
  };
  const uint32_t input_size = input.size();

  // Initialize pointer to quantization parameters.
  QuantizationParametersT quantization_params = QuantizationParametersT();

  FillSingleMinMax(input.data(), input_size, &quantization_params);
  const std::vector<float> expected_min_max = {
      13, 40,  // min max
  };
  EXPECT_EQ(quantization_params.min.size(), 1);
  EXPECT_EQ(quantization_params.max.size(), 1);
  EXPECT_EQ(quantization_params.min[0], expected_min_max[0]);
  EXPECT_EQ(quantization_params.max[0], expected_min_max[1]);
}

// kMaxQuantizedValue should match that of quantization_utils.cc.
TEST_F(QuantizationUtilsTest, GetSymmetricScalesFromMaxMin) {
  const int8_t kMaxQuantizedValue = 127;
  tflite::TestErrorReporter error_reporter_;
  // Create data.
  auto quantization = std::make_unique<QuantizationParametersT>();
  quantization->min = {-0.00001, -7.0, -2.0};
  quantization->max = {0.00001, 1.0, -1.0};
  std::vector<float> scales = std::vector<float>(quantization->min.size());

  GetSymmetricScalesFromMaxMin(quantization.get(), &scales, &error_reporter_);
  const std::vector<float> expected_scales = {0.00001 / kMaxQuantizedValue,
                                              7.0 / kMaxQuantizedValue,
                                              2.0 / kMaxQuantizedValue};
  EXPECT_EQ(scales, expected_scales);
}

// kMaxQuantizedValue should match that of quantization_utils.cc
TEST_F(QuantizationUtilsTest, AdjustWeightScaleForBiasPerChannel) {
  // 2^(31) = 2147483648
  const int32_t kScale = std::numeric_limits<int32_t>::max();
  const int8_t kMaxQuantizedValue = 127;
  tflite::TestErrorReporter error_reporter_;

  // Create data.
  auto quant_params = std::make_unique<QuantizationParametersT>();
  const float small_val = 0.0000001;
  const std::vector<float> orig_mins = {-small_val, -7.0};
  quant_params->min = orig_mins;
  const std::vector<float> orig_maxs = {small_val, 1.0};
  quant_params->max = orig_maxs;
  std::vector<float> scales = std::vector<float>(quant_params->min.size());

  GetSymmetricScalesFromMaxMin(quant_params.get(), &scales, &error_reporter_);
  const std::vector<float> expected_scales = {small_val / kMaxQuantizedValue,
                                              7.0 / kMaxQuantizedValue};
  EXPECT_EQ(scales, expected_scales);

  const float input_scale = 0.05;

  // Initialize bias.
  float bias_data[] = {4.0, 4.0};
  const size_t bias_size = 2;
  // Quantized bias would be {101600000000, -- }.

  AdjustWeightsForBiasScale(quant_params.get(), bias_data, bias_size,
                            input_scale, &error_reporter_);

  std::vector<float> new_scales = std::vector<float>(quant_params->min.size());
  GetSymmetricScalesFromMaxMin(quant_params.get(), &new_scales,
                               &error_reporter_);

  // Adjust min and max for first channel.
  EXPECT_TRUE(new_scales[0] > scales[0]);
  EXPECT_TRUE(std::abs(bias_data[0]) / kScale <=
              0.6 * input_scale * new_scales[0]);
  EXPECT_TRUE(orig_mins[0] > quant_params->min[0]);
  EXPECT_TRUE(orig_maxs[0] < quant_params->max[0]);
  // No change for second channel.
  EXPECT_TRUE(std::abs(bias_data[1]) / kScale <=
              0.6 * input_scale * new_scales[1]);
  EXPECT_EQ(orig_mins[1], quant_params->min[1]);
  EXPECT_EQ(orig_maxs[1], quant_params->max[1]);
}

// kMaxQuantizedValue should match that of quantization_utils.cc.
TEST_F(QuantizationUtilsTest, AdjustWeightScaleForBiasPerLayer) {
  // 2^(31) = 2147483648
  const int32_t kScale = std::numeric_limits<int32_t>::max();
  const int8_t kMaxQuantizedValue = 127;
  tflite::TestErrorReporter error_reporter_;

  // Create data.
  auto quant_params = std::make_unique<QuantizationParametersT>();
  float small_val = 0.0000001;
  const std::vector<float> orig_mins = {-small_val};
  quant_params->min = orig_mins;
  const std::vector<float> orig_maxs = {small_val};
  quant_params->max = orig_maxs;
  std::vector<float> scales = std::vector<float>(quant_params->min.size());
  GetSymmetricScalesFromMaxMin(quant_params.get(), &scales, &error_reporter_);
  const std::vector<float> expected_scales = {small_val / kMaxQuantizedValue};
  EXPECT_EQ(scales, expected_scales);

  const float input_scale = 0.05;

  // Initialize bias.
  float bias_data[] = {4.0};
  const size_t bias_size = 1;
  // Quantized bias would be {101600000000}.

  AdjustWeightsForBiasScale(quant_params.get(), bias_data, bias_size,
                            input_scale, &error_reporter_);

  std::vector<float> new_scales = std::vector<float>(quant_params->min.size());
  GetSymmetricScalesFromMaxMin(quant_params.get(), &new_scales,
                               &error_reporter_);

  // Adjust min and max.
  EXPECT_TRUE(new_scales[0] > scales[0]);
  EXPECT_TRUE(std::abs(bias_data[0]) / kScale <=
              0.6 * input_scale * new_scales[0]);
  EXPECT_TRUE(orig_mins[0] > quant_params->min[0]);
  EXPECT_TRUE(orig_maxs[0] < quant_params->max[0]);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensorFromMinMax) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();

  // The test model has the incorrect number of min and max values for per-layer
  // quantization.
  weights_tensor->quantization->min =
      std::vector<float>(1, weights_tensor->quantization->min[0]);
  weights_tensor->quantization->max =
      std::vector<float>(1, weights_tensor->quantization->max[0]);

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  bool per_channel = false;
  int per_axis_index = 0;
  tflite::TestErrorReporter error_reporter_;
  EXPECT_EQ(QuantizeWeight(&model, weights_tensor, per_channel, per_axis_index,
                           &error_reporter_),
            kTfLiteOk);

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);
  EXPECT_EQ(quant_buffer_size * 4, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensorNullInputs) {
  tflite::TestErrorReporter error_reporter_;
  EXPECT_EQ(SymmetricQuantizeTensor(nullptr, nullptr), kTfLiteError);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensorNullQuantParams) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();
  // Empty quantization parameters.
  weights_tensor->quantization = std::make_unique<QuantizationParametersT>();

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  EXPECT_EQ(SymmetricQuantizeTensor(&model, weights_tensor), kTfLiteOk);

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);
  EXPECT_EQ(quant_buffer_size * 4, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeTensor) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  EXPECT_EQ(SymmetricQuantizeTensor(&model, weights_tensor), kTfLiteOk);

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);
  EXPECT_EQ(quant_buffer_size * 4, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, QuantizeFloat16Clamp) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  constexpr int kNumElements = 6;
  const std::vector<float> weights = {2.0, 1.0, 65504., 65505, -65504., -99999};
  auto weights_reinterpreted_data =
      reinterpret_cast<const unsigned char*>(weights.data());
  buffer->data.assign(weights_reinterpreted_data,
                      weights_reinterpreted_data + weights.size() * 4);
  tensor->buffer = 0;
  tensor->shape = {1, kNumElements};

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  EXPECT_EQ(
      QuantizeTensorFloat16(model.get(), model->subgraphs[0]->tensors[0].get()),
      kTfLiteOk);
  auto weightsf16 = reinterpret_cast<Eigen::half*>(
      model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data.data());
  std::vector<float> wf32(kNumElements);
  std::transform(weightsf16, weightsf16 + 6, wf32.begin(),
                 [](Eigen::half a) { return static_cast<float>(a); });

  EXPECT_THAT(wf32,
              ElementsAreArray({2.0, 1.0, 65504., 65504., -65504., -65504.}));
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_FLOAT16);
}

TEST_F(QuantizationUtilsTest, QuantizeFloat16) {
  // Conv model has weights between 0 and 10.
  // Quantize the weights tensor.
  ASSERT_TRUE(g_test_model_dir != nullptr);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  auto conv_op = subgraph->operators.at(0).get();
  ASSERT_EQ(
      GetBuiltinCode(model.operator_codes.at(conv_op->opcode_index).get()),
      BuiltinOperator_CONV_2D);
  int32_t weights_tensor_idx = conv_op->inputs[1];
  TensorT* weights_tensor = subgraph->tensors.at(weights_tensor_idx).get();

  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT32);
  size_t float_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();

  EXPECT_EQ(QuantizeTensorFloat16(&model, weights_tensor), kTfLiteOk);

  size_t quant_buffer_size =
      model.buffers.at(weights_tensor->buffer)->data.size();
  EXPECT_EQ(weights_tensor->type, TensorType_FLOAT16);
  EXPECT_EQ(quant_buffer_size * 2, float_buffer_size);
}

TEST_F(QuantizationUtilsTest, AddQuantizationParams) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  const std::vector<float> scales = {0.5, 1.0, 1.5};
  const std::vector<int64_t> zero_points = {5, 10, 15};
  const int32_t quantizated_dimension = 3;
  const std::vector<uint8_t> buffer_data = {1, 2, 3, 4};
  const int32_t buffer_size = 4;
  tensor->buffer = 0;

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  EXPECT_EQ(AddQuantizationParams(
                scales, zero_points, quantizated_dimension, buffer_data.data(),
                buffer_size, TensorType_INT8, model.get(),
                model->subgraphs[0]->tensors[0].get(), &error_reporter_),
            kTfLiteOk);
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->scale,
              ElementsAreArray(scales));
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->zero_point,
              ElementsAreArray(zero_points));
  EXPECT_THAT(model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data,
              ElementsAreArray(buffer_data));
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT8);
}

TEST_F(QuantizationUtilsTest, SymmetricQuantizeFloatsToInt16Test) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  const float weight_scale = 0.5;
  const float input_scale = 0.5;
  std::vector<float> layer_norm_data = {4.0, 1.0, -1.0, 8.0};
  auto layer_norm_reinterpreted_data =
      reinterpret_cast<const unsigned char*>(layer_norm_data.data());
  buffer->data.assign(
      layer_norm_reinterpreted_data,
      layer_norm_reinterpreted_data + layer_norm_data.size() * 4);
  tensor->buffer = 0;
  tensor->shape = {4};
  tensor->quantization = std::make_unique<QuantizationParametersT>();

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  EXPECT_EQ(SymmetricQuantizeFloatsToInt16(
                model.get(), model->subgraphs[0]->tensors[0].get(),
                input_scale * weight_scale, &error_reporter_),
            kTfLiteOk);

  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->scale[0],
              weight_scale * input_scale);
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->zero_point[0], 0);

  auto result = reinterpret_cast<int16_t*>(
      model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data.data());
  EXPECT_EQ(result[0], 16);
  EXPECT_EQ(result[1], 4);
  EXPECT_EQ(result[2], -4);
  EXPECT_EQ(result[3], 32);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT16);
}

TEST_F(QuantizationUtilsTest, SymmetricPerLayerBiasQuantize) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  const float weight_scale = 0.5;
  const float input_scale = 0.5;
  std::vector<float> bias_data = {4.0, 1.0};
  auto bias_reinterpreted_data =
      reinterpret_cast<const unsigned char*>(bias_data.data());
  buffer->data.assign(bias_reinterpreted_data,
                      bias_reinterpreted_data + bias_data.size() * 4);
  tensor->buffer = 0;
  tensor->shape = {2, 1, 1, 1};
  tensor->quantization = std::make_unique<QuantizationParametersT>();

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  EXPECT_EQ(SymmetricPerLayerBiasQuantize<int32_t>(
                model.get(), model->subgraphs[0]->tensors[0].get(),
                input_scale * weight_scale, &error_reporter_),
            kTfLiteOk);

  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->scale[0],
              weight_scale * input_scale);
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->zero_point[0], 0);

  const uint32_t* d1 = reinterpret_cast<const uint32_t*>(
      model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data.data());
  EXPECT_EQ(d1[0], 0x00000010);
  EXPECT_EQ(d1[1], 0x00000004);

  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT32);
}

TEST_F(QuantizationUtilsTest, GetEffectiveScale) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto op = std::make_unique<OperatorT>();
  tensor->quantization = std::make_unique<QuantizationParametersT>();
  tensor->quantization->scale.push_back(3.0);
  op->inputs.push_back(0);

  // Wire the model.
  subgraph->operators.push_back(std::move(op));
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));

  // Call and verify.
  EXPECT_EQ(GetEffectiveScale(model.get(), model->subgraphs[0].get(), 0, {0},
                              {}, {5.0}),
            15.0);
}

TEST_F(QuantizationUtilsTest, SymmetricPerChannelBiasQuantize) {
  // Create data.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto tensor = std::make_unique<TensorT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  const std::vector<float> weight_scales = {0.5, 1.0};
  const float input_scale = 0.5;
  std::vector<float> bias_data = {4.0, 1.0};
  auto bias_reinterpreted_data =
      reinterpret_cast<const unsigned char*>(bias_data.data());
  buffer->data.assign(bias_reinterpreted_data,
                      bias_reinterpreted_data + bias_data.size() * 4);
  tensor->buffer = 0;
  tensor->shape = {2, 1, 1, 1};
  tensor->quantization = std::make_unique<QuantizationParametersT>();

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  EXPECT_EQ(SymmetricPerChannelBiasQuantize<int32_t>(
                model.get(), model->subgraphs[0]->tensors[0].get(), input_scale,
                weight_scales.data(), 2, &error_reporter_),
            kTfLiteOk);

  const uint32_t* d1 = reinterpret_cast<const uint32_t*>(
      model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data.data());
  EXPECT_EQ(d1[0], 0x00000010);
  EXPECT_EQ(d1[1], 0x00000002);

  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT32);
}

TEST_F(QuantizationUtilsTest, ExtendToPowerOfTwo) {
  EXPECT_EQ(GetPowerOfTwoScale(-1.0, 1.0), 0);
  EXPECT_EQ(GetPowerOfTwoScale(-10.0, 10.0), 4);
  EXPECT_EQ(GetPowerOfTwoScale(3.0, 10.0), 4);
}

}  // namespace
}  // namespace utils
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  tensorflow::string model_file;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_model_file", &model_file,
                       "Path to test tflite model file."),
  };

  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << "Required test_model_file\n";
    std::abort();
  }
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
