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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
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

TEST(QuantizationUtilsTest, NumElements) {
  TensorT tensor;
  tensor.shape = {1, 2, 3, 4};
  uint64_t num_elements;
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 1 * 2 * 3 * 4);

  tensor.shape = {5};
  EXPECT_EQ(kTfLiteOk, NumElements(tensor, &num_elements));
  EXPECT_EQ(num_elements, 5);

  tensor.shape = {};
  EXPECT_EQ(kTfLiteError, NumElements(tensor, &num_elements));
}

TEST(QuantizationUtilsTest, GetAsymmetricQuantizationParamsUnitRange) {
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

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithAllPositiveRange) {
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

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithAllNegativeRange) {
  // The min should get nudged to include 0, so the effective range is [-6, 0].
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

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithZeroInRange) {
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

TEST(QuantizationUtilsTest, AsymmetricQuantizationParamsWithZeroMinMax) {
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

TEST(QuantizationUtilsTest, SymmetricPerChannelQuantization) {
  // Set up an input with [3, 2, 2, 2] size and 0 is the channel index.
  const std::vector<float> input = {
      3.0, 2.0, 5.0,  -2.0, 3.0,  2.0,  5.0,  -2.0,  // Channel 1.
      1.0, 2.0, 3.0,  4.0,  5.0,  6.0,  7.0,  8.0,   // Channel 2.
      1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,  // Channel 3.
  };
  const std::vector<int32_t> dimension = {3, 2, 2, 2};
  const int channel_index = 0;

  // Create holder for output scale and data.
  std::vector<float> output_scales(3);
  std::vector<int8_t> output_data(3 * 2 * 2 * 2);

  // Call SymmetricPerChannelQuantization and verify the result.
  SymmetricPerChannelQuantization(input.data(), dimension, channel_index,
                                  &output_scales, &output_data);
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

TEST(QuantizationUtilsTest, SymmetricPerChannelQuantizeValues) {
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

TEST(QuantizationUtilsTest, SymmetricQuantizeTensorNullInputs) {
  EXPECT_EQ(SymmetricQuantizeTensor(nullptr, nullptr), kTfLiteError);
}

TEST(QuantizationUtilsTest, SymmetricQuantizeTensor) {
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
  ASSERT_EQ(model.operator_codes.at(conv_op->opcode_index)->builtin_code,
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

TEST(QuantizationUtilsTest, AddQuantizationParams) {
  // Create data.
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto tensor = absl::make_unique<TensorT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
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
  EXPECT_EQ(
      AddQuantizationParams(scales, zero_points, quantizated_dimension,
                            buffer_data.data(), buffer_size, TensorType_INT8,
                            model.get(), model->subgraphs[0]->tensors[0].get()),
      kTfLiteOk);
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->scale,
              ElementsAreArray(scales));
  EXPECT_THAT(model->subgraphs[0]->tensors[0]->quantization->zero_point,
              ElementsAreArray(zero_points));
  EXPECT_THAT(model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data,
              ElementsAreArray(buffer_data));
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT8);
}

TEST(QuantizationUtilsTest, SymmetricPerChannelBiasQuantize) {
  // Create data.
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto tensor = absl::make_unique<TensorT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
  const std::vector<float> weight_scales = {0.5, 1.0};
  const float input_scale = 0.5;
  std::vector<float> bias_data = {4.0, 1.0};
  auto bias_reinterpreted_data =
      reinterpret_cast<const unsigned char*>(bias_data.data());
  buffer->data.assign(bias_reinterpreted_data,
                      bias_reinterpreted_data + bias_data.size() * 4);
  tensor->buffer = 0;
  tensor->shape = {2, 1, 1, 1};
  tensor->quantization = absl::make_unique<QuantizationParametersT>();

  // Wire the model.
  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->buffers.push_back(std::move(buffer));

  // Call and verify.
  EXPECT_EQ(SymmetricPerChannelBiasQuantize(
                model.get(), model->subgraphs[0]->tensors[0].get(), input_scale,
                weight_scales.data(), 2, 0),
            kTfLiteOk);
  EXPECT_THAT(model->buffers[model->subgraphs[0]->tensors[0]->buffer]->data,
              ElementsAreArray({16, 0, 0, 0, 2, 0, 0, 0}));
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_INT32);
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
