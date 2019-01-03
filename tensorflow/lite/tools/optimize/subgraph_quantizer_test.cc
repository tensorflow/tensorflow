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
#include "tensorflow/lite/tools/optimize/subgraph_quantizer.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/symmetric_per_channel_params.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace internal {
namespace {

std::unique_ptr<FlatBufferModel> ReadTestModel1() {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir,
                                             kModelWithMinus128Plus127Weights);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModel> ReadTestModel2() {
  auto model_path =
      tensorflow::io::JoinPath(*g_test_model_dir, kModelWith0Plus10Weights);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

TEST(SubgraphQuantizerTest, VerifyConvQuantizationWithUnitScale) {
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadTestModel1();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  FailOnErrorReporter error_reporter;
  SubgraphQuantizer quantizer(&model, subgraph, &error_reporter);
  auto status = quantizer.QuantizeOperator(0);
  ASSERT_EQ(kTfLiteOk, status);

  auto conv_op = subgraph->operators[0].get();
  const int input_tensor_idx = 0;
  const int weights_tensor_idx = 1;
  const int bias_tensor_index = 2;
  const int output_tensor_idx = 0;
  const auto bias_tensor =
      subgraph->tensors[conv_op->inputs[bias_tensor_index]].get();
  const auto input_tensor =
      subgraph->tensors[conv_op->inputs[input_tensor_idx]].get();
  const auto weights_tensor =
      subgraph->tensors[conv_op->inputs[weights_tensor_idx]].get();
  const auto output_tensor =
      subgraph->tensors[conv_op->outputs[output_tensor_idx]].get();

  EXPECT_EQ(bias_tensor->type, TensorType_INT32);
  EXPECT_EQ(input_tensor->type, TensorType_INT8);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  const int out_channel_size = weights_tensor->shape[0];
  std::unique_ptr<SymmetricPerChannelParams> bias_params;
  std::unique_ptr<SymmetricPerChannelParams> weights_params;
  ASSERT_EQ(kTfLiteOk, SymmetricPerChannelParams::ReadFromTensor(
                           *weights_tensor, &weights_params));
  ASSERT_EQ(kTfLiteOk, SymmetricPerChannelParams::ReadFromTensor(*bias_tensor,
                                                                 &bias_params));

  ASSERT_EQ(bias_params->scales().size(), out_channel_size);
  ASSERT_EQ(weights_params->scales().size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(
        bias_params->scales()[i],
        input_tensor->quantization->scale[0] * weights_params->scales()[i],
        eps);
  }
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_EQ(weights_params->scales()[i], 1);
    EXPECT_EQ(bias_params->scales()[i], 1);
  }
  EXPECT_EQ(input_tensor->quantization->scale[0], 1);
  EXPECT_EQ(output_tensor->quantization->scale[0], 1);

  const auto bias_buffer = model.buffers[bias_tensor->buffer].get();
  ASSERT_EQ(bias_buffer->data.size(), sizeof(int32_t) * bias_tensor->shape[0]);
  const int32_t* bias_values =
      reinterpret_cast<int32_t*>(bias_buffer->data.data());
  const auto original_bias_buffer =
      readonly_model->buffers()->Get(bias_tensor->buffer);
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  for (size_t i = 0; i < bias_tensor->shape[0]; i++) {
    auto dequantized_value = bias_values[i] * bias_params->scales()[i];
    EXPECT_NEAR(dequantized_value, bias_float_buffer[i], eps);
  }

  const auto weights_buffer = model.buffers[weights_tensor->buffer].get();
  const auto original_weights_buffer =
      readonly_model->buffers()->Get(weights_tensor->buffer);
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto dequantized_value =
          weight_values[element_idx] * weights_params->scales()[channel_idx];
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx], eps);
    }
  }
}

TEST(SubgraphQuantizerTest, VerifyConvQuantization) {
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadTestModel2();
  ASSERT_TRUE(test_model);
  auto readonly_model = test_model->GetModel();
  ASSERT_TRUE(readonly_model);
  ASSERT_TRUE(readonly_model->subgraphs());
  ASSERT_GE(readonly_model->subgraphs()->size(), 1);
  tflite::ModelT model;
  readonly_model->UnPackTo(&model);
  auto subgraph = model.subgraphs[0].get();
  FailOnErrorReporter error_reporter;
  SubgraphQuantizer quantizer(&model, subgraph, &error_reporter);
  auto status = quantizer.QuantizeOperator(0);
  ASSERT_EQ(kTfLiteOk, status);

  auto conv_op = subgraph->operators[0].get();
  const int input_tensor_idx = 0;
  const int weights_tensor_idx = 1;
  const int bias_tensor_index = 2;
  const int output_tensor_idx = 0;
  const auto bias_tensor =
      subgraph->tensors[conv_op->inputs[bias_tensor_index]].get();
  const auto input_tensor =
      subgraph->tensors[conv_op->inputs[input_tensor_idx]].get();
  const auto weights_tensor =
      subgraph->tensors[conv_op->inputs[weights_tensor_idx]].get();
  const auto output_tensor =
      subgraph->tensors[conv_op->outputs[output_tensor_idx]].get();

  EXPECT_EQ(bias_tensor->type, TensorType_INT32);
  EXPECT_EQ(input_tensor->type, TensorType_INT8);
  EXPECT_EQ(weights_tensor->type, TensorType_INT8);

  ASSERT_TRUE(weights_tensor->quantization);
  const int out_channel_size = weights_tensor->shape[0];
  std::unique_ptr<SymmetricPerChannelParams> bias_params;
  std::unique_ptr<SymmetricPerChannelParams> weights_params;
  ASSERT_EQ(kTfLiteOk, SymmetricPerChannelParams::ReadFromTensor(
                           *weights_tensor, &weights_params));
  ASSERT_EQ(kTfLiteOk, SymmetricPerChannelParams::ReadFromTensor(*bias_tensor,
                                                                 &bias_params));

  ASSERT_EQ(bias_params->scales().size(), out_channel_size);
  ASSERT_EQ(weights_params->scales().size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(
        bias_params->scales()[i],
        input_tensor->quantization->scale[0] * weights_params->scales()[i],
        eps);
  }

  const auto bias_buffer = model.buffers[bias_tensor->buffer].get();
  ASSERT_EQ(bias_buffer->data.size(), sizeof(int32_t) * bias_tensor->shape[0]);
  const int32_t* bias_values =
      reinterpret_cast<int32_t*>(bias_buffer->data.data());
  const auto original_bias_buffer =
      readonly_model->buffers()->Get(bias_tensor->buffer);
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  for (size_t i = 0; i < bias_tensor->shape[0]; i++) {
    auto scale = bias_params->scales()[i];
    auto dequantized_value = bias_values[i] * scale;
    EXPECT_NEAR(dequantized_value, bias_float_buffer[i], scale / 2);
  }

  const auto weights_buffer = model.buffers[weights_tensor->buffer].get();
  const auto original_weights_buffer =
      readonly_model->buffers()->Get(weights_tensor->buffer);
  const int8_t* weight_values =
      reinterpret_cast<int8_t*>(weights_buffer->data.data());
  const float* weights_float_buffer =
      reinterpret_cast<const float*>(original_weights_buffer->data()->data());
  ASSERT_EQ(sizeof(float) * weights_buffer->data.size(),
            original_weights_buffer->data()->size());
  int num_values_in_channel = weights_buffer->data.size() / out_channel_size;
  for (size_t channel_idx = 0; channel_idx < out_channel_size; channel_idx++) {
    for (size_t j = 0; j < num_values_in_channel; j++) {
      size_t element_idx = channel_idx * out_channel_size + j;
      auto scale = weights_params->scales()[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx],
                  scale / 2);
    }
  }
}

}  // namespace
}  // namespace internal
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  tensorflow::string model_file;
  const std::vector<tensorflow::Flag> flag_list = {
      tensorflow::Flag("test_model_file", &model_file,
                       "Path to test tflite model file."),
  };

  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  CHECK(parse_result) << "Required test_model_file";
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
