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
#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/tools/optimize/subgraph_quantizer.h"
#include "tensorflow/lite/tools/optimize/test_util.h"

namespace {
tensorflow::string* g_test_model_dir = nullptr;
}  // namespace

namespace tflite {
namespace optimize {
namespace internal {
namespace {

std::unique_ptr<FlatBufferModel> ReadModel(const char* model) {
  auto model_path = tensorflow::io::JoinPath(*g_test_model_dir, model);
  return FlatBufferModel::BuildFromFile(model_path.c_str());
}

std::unique_ptr<FlatBufferModel> ReadConvModel1() {
  return ReadModel(kConvModelWithMinus128Plus127Weights);
}

std::unique_ptr<FlatBufferModel> ReadConvModel2() {
  return ReadModel(kConvModelWith0Plus10Weights);
}

std::unique_ptr<FlatBufferModel> ReadSoftmaxModel() {
  return ReadModel(kSingleSoftmaxModelMinMinus5MaxPlus5);
}

std::unique_ptr<FlatBufferModel> ReadAvgPoolModel() {
  return ReadModel(kSingleAvgPoolModelMinMinus5MaxPlus5);
}

TEST(SubgraphQuantizerTest, VerifyConvQuantizationWithUnitScale) {
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel1();
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
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;

  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;

  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);


  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_EQ(weights_scales[i], 1);
    EXPECT_EQ(bias_scales[i], 1);
    EXPECT_EQ(weights_zero_points[i], 0);
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

  const float eps = 1e-7;
  for (size_t i = 0; i < bias_tensor->shape[0]; i++) {
    const float bias_scale =
        input_tensor->quantization->scale[0] * weights_scales[i];
    auto dequantized_value = bias_values[i] * bias_scale;
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
          weight_values[element_idx] * weights_scales[channel_idx];
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx], eps);
    }
  }
}

TEST(SubgraphQuantizerTest, VerifyConvQuantization) {
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadConvModel2();
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
  ASSERT_TRUE(bias_tensor->quantization);
  ASSERT_TRUE(weights_tensor->quantization);
  const std::vector<float>& bias_scales = bias_tensor->quantization->scale;
  const std::vector<float>& weights_scales =
      weights_tensor->quantization->scale;
  const std::vector<int64_t>& weights_zero_points =
      weights_tensor->quantization->zero_point;

  ASSERT_EQ(bias_scales.size(), out_channel_size);
  ASSERT_EQ(weights_scales.size(), out_channel_size);
  ASSERT_EQ(weights_zero_points.size(), out_channel_size);
  ASSERT_EQ(input_tensor->quantization->scale.size(), 1);
  ASSERT_EQ(output_tensor->quantization->scale.size(), 1);

  const float eps = 1e-7;

  // Bias scale should be input * per_channel_weight_scale.
  for (size_t i = 0; i < out_channel_size; i++) {
    EXPECT_NEAR(bias_scales[i],
                input_tensor->quantization->scale[0] * weights_scales[i], eps);
  }

  const auto bias_buffer = model.buffers[bias_tensor->buffer].get();
  ASSERT_EQ(bias_buffer->data.size(), sizeof(int32_t) * bias_tensor->shape[0]);
  const int32_t* bias_values =
      reinterpret_cast<int32_t*>(bias_buffer->data.data());
  const auto original_bias_buffer =
      readonly_model->buffers()->Get(bias_tensor->buffer);
  const float* bias_float_buffer =
      reinterpret_cast<const float*>(original_bias_buffer->data()->data());

  for (size_t i = 0; i < out_channel_size; i++) {
    auto dequantized_value = bias_values[i] * bias_scales[i];
    EXPECT_NEAR(dequantized_value, bias_float_buffer[i], bias_scales[i] / 2);
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
      auto scale = weights_scales[channel_idx];
      auto zero_point = weights_zero_points[channel_idx];
      auto dequantized_value = weight_values[element_idx] * scale;
      EXPECT_NEAR(dequantized_value, weights_float_buffer[element_idx],
                  scale / 2);
      EXPECT_EQ(zero_point, 0);
    }
  }
}

void VerifyAsymmetricQuantizationScale(
    const QuantizationParameters& float_quant_params,
    const QuantizationParametersT& quantized_quant_params) {
  const float eps = 1e-7;
  ASSERT_EQ(float_quant_params.min()->size(), 1);
  ASSERT_EQ(float_quant_params.max()->size(), 1);
  float float_min = std::min(0.f, float_quant_params.min()->Get(0));
  float float_max = std::max(0.f, float_quant_params.max()->Get(0));

  ASSERT_EQ(quantized_quant_params.scale.size(), 1);
  ASSERT_EQ(quantized_quant_params.zero_point.size(), 1);

  float scale = (float_max - float_min) / 255;
  EXPECT_NEAR(scale, quantized_quant_params.scale[0], eps);
}

TEST(SubgraphQuantizerTest, VerifySoftmaxQuantization) {
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadSoftmaxModel();
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

  auto op = subgraph->operators[0].get();
  // Model has a single softmax op.
  ASSERT_EQ(op->opcode_index, 0);
  ASSERT_EQ(model.operator_codes[0].get()->builtin_code,
            BuiltinOperator_SOFTMAX);

  ASSERT_EQ(op->inputs.size(), 1);
  ASSERT_EQ(op->outputs.size(), 1);
  auto float_graph = readonly_model->subgraphs()->Get(0);

  // Verify input.
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  // Verify output.
  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);

  ASSERT_EQ(output_quant_params->scale.size(), 1);
  ASSERT_EQ(output_quant_params->zero_point.size(), 1);
  ASSERT_EQ(1.0f / 256.0f, output_quant_params->scale[0]);
  ASSERT_EQ(-128, output_quant_params->zero_point[0]);
}

TEST(SubgraphQuantizerTest, VerifyAvgPoolQuantization) {
  ASSERT_TRUE(g_test_model_dir);
  ASSERT_FALSE(g_test_model_dir->empty());
  auto test_model = ReadAvgPoolModel();
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

  auto op = subgraph->operators[0].get();
  // Model has a single AveragePool op.
  ASSERT_EQ(op->opcode_index, 0);
  ASSERT_EQ(model.operator_codes[0].get()->builtin_code,
            BuiltinOperator_AVERAGE_POOL_2D);

  ASSERT_EQ(op->inputs.size(), 1);
  ASSERT_EQ(op->outputs.size(), 1);

  auto float_graph = readonly_model->subgraphs()->Get(0);
  ASSERT_EQ(float_graph->tensors()->Get(op->inputs[0])->type(),
            TensorType_FLOAT32);
  ASSERT_EQ(float_graph->tensors()->Get(op->outputs[0])->type(),
            TensorType_FLOAT32);

  EXPECT_EQ(subgraph->tensors[op->inputs[0]].get()->type, TensorType_INT8);
  EXPECT_EQ(subgraph->tensors[op->outputs[0]].get()->type, TensorType_INT8);

  auto float_input_quant_params =
      float_graph->tensors()->Get(op->inputs[0])->quantization();
  auto input_quant_params =
      subgraph->tensors[op->inputs[0]]->quantization.get();
  VerifyAsymmetricQuantizationScale(*float_input_quant_params,
                                    *input_quant_params);

  auto float_output_quant_params =
      float_graph->tensors()->Get(op->outputs[0])->quantization();
  auto output_quant_params =
      subgraph->tensors[op->outputs[0]]->quantization.get();
  ASSERT_EQ(float_output_quant_params->min()->size(), 1);
  ASSERT_EQ(float_output_quant_params->max()->size(), 1);
  ASSERT_EQ(output_quant_params->min.size(), 1);
  ASSERT_EQ(output_quant_params->max.size(), 1);

  // Make sure the input min/maxes are propagated to outputs.
  EXPECT_EQ(input_quant_params->min[0], output_quant_params->min[0]);
  EXPECT_EQ(input_quant_params->max[0], output_quant_params->max[0]);
  EXPECT_EQ(input_quant_params->scale[0], output_quant_params->scale[0]);
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
  if (!parse_result) {
    std::cerr << "Required test_model_file\n";
    std::abort();
  }
  g_test_model_dir =
      new tensorflow::string(tensorflow::io::Dirname(model_file));
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);
  return RUN_ALL_TESTS();
}
