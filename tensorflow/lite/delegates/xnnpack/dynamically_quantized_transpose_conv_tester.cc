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

#include "tensorflow/lite/delegates/xnnpack/dynamically_quantized_transpose_conv_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void DynamicallyQuantizedTransposeConvTester::Test(
    TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10, 10), rng);

  const std::vector<int8_t> kernel_data = GenerateKernelData();
  const std::vector<float> bias_data = GenerateBiasData();
  const std::vector<float> kernel_scale_data = GenerateKernelScaleData();
  std::vector<char> drq_buffer =
      CreateDRQTfLiteModel(kernel_data, bias_data, kernel_scale_data);
  std::vector<char> dequantize_buffer =
      CreateDequantizeTfLiteModel(kernel_data, bias_data, kernel_scale_data);
  const Model* drq_model = GetModel(drq_buffer.data());
  const Model* dequantize_model = GetModel(dequantize_buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          drq_model, ::tflite::ops::builtin::BuiltinOpResolverWithXNNPACK())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          dequantize_model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  if (weights_cache_ != nullptr) {
    TfLiteXNNPackDelegateWeightsCacheFinalizeHard(weights_cache_);
  }

  const int input_data_size =
      BatchSize() * InputHeight() * InputWidth() * InputChannels();
  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate_n(default_input_data, input_data_size, std::ref(f32rng));

  float* xnnpack_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy_n(default_input_data, input_data_size, xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data = default_interpreter->typed_tensor<float>(
      default_interpreter->outputs()[0]);
  float* xnnpack_output_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->outputs()[0]);

  const int num_output_values =
      BatchSize() * OutputHeight() * OutputWidth() * OutputChannels();
  int different_output_values = 0;
  for (size_t i = 0; i < num_output_values; i++) {
    if (std::abs(default_output_data[i] - xnnpack_output_data[i]) >
        0.1 * std::abs(default_output_data[i])) {
      ++different_output_values;
    }
  }

  if (different_output_values > 0.05 * num_output_values) {
    GTEST_FAIL() << (float)different_output_values / num_output_values * 100.f
                 << "% of output values differ";
  }
}

std::vector<int8_t>
DynamicallyQuantizedTransposeConvTester::GenerateKernelData() const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto range_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                 std::numeric_limits<int8_t>::min(),
                                 std::numeric_limits<int8_t>::max()),
                             std::ref(rng));
  std::vector<int8_t> filter_data(OutputChannels() * KernelHeight() *
                                  KernelWidth() * InputChannels());

  for (int32_t oc = 0; oc < OutputChannels(); oc++) {
    // Use the same range of all-positive or all-negative values to generate
    // all weights within the same output channel, but different ranges for
    // different output channels. This ensures that no catastrophic
    // cancellation occur, but test covers both positive and negative
    // inputs.
    const int32_t range = range_rng();
    const auto value_dist = std::uniform_int_distribution<int32_t>(
        std::min(range, 0), std::max(range, 0));
    auto value_rng = std::bind(value_dist, std::ref(rng));
    for (int32_t ic = 0; ic < InputChannels(); ic++) {
      for (int32_t y = 0; y < KernelHeight(); y++) {
        for (int32_t x = 0; x < KernelWidth(); x++) {
          const int32_t index =
              ((oc * KernelHeight() + y) * KernelWidth() + x) *
                  InputChannels() +
              ic;
          filter_data[index] = value_rng();
        }
      }
    }
  }
  return filter_data;
}

std::vector<float> DynamicallyQuantizedTransposeConvTester::GenerateBiasData()
    const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto bias_rng =
      std::bind(std::uniform_real_distribution<float>(-10, 10), std::ref(rng));
  std::vector<float> bias_data(OutputChannels());
  std::generate(bias_data.begin(), bias_data.end(), std::ref(bias_rng));
  return bias_data;
}

std::vector<float>
DynamicallyQuantizedTransposeConvTester::GenerateKernelScaleData() const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto kernel_scale_rng =
      std::bind(std::uniform_real_distribution<float>(0.1, 3), std::ref(rng));

  std::vector<float> kernel_scale(OutputChannels());
  std::generate(kernel_scale.begin(), kernel_scale.end(),
                std::ref(kernel_scale_rng));

  return kernel_scale;
}

std::vector<char> DynamicallyQuantizedTransposeConvTester::CreateDRQTfLiteModel(
    const std::vector<int8_t>& filter_data, const std::vector<float>& bias_data,
    const std::vector<float>& kernel_scale) const {
  /*************************** Define operator codes **************************/
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_TRANSPOSE_CONV)}};

  /****************************** Define buffers ******************************/
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({}))}};
  const int filter_buffer_id = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(filter_data.data()),
                           sizeof(int8_t) * filter_data.size())));
  const int bias_buffer_id = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(bias_data.data()),
                           sizeof(float) * bias_data.size())));
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), OutputChannels()}};
  const int output_shape_buffer_id = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder, builder.CreateVector(
                   reinterpret_cast<const uint8_t*>(output_shape.data()),
                   sizeof(int32_t) * output_shape.size())));

  /****************************** Define tensors ******************************/
  const std::vector<int32_t> filter_shape = {OutputChannels(), KernelHeight(),
                                             KernelWidth(), InputChannels()};
  const std::vector<int32_t> bias_shape = {OutputChannels()};
  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  const int input_tensor_id = tensors.size();
  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), InputChannels()}};
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
      TensorType_FLOAT32));

  const int filter_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
      TensorType_INT8,
      /*buffer=*/filter_buffer_id, /*name=*/0,
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>(kernel_scale),
          builder.CreateVector<int64_t>(
              std::vector<int64_t>(OutputChannels(), 0)))));

  const int bias_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(bias_shape.data(), bias_shape.size()),
      TensorType_FLOAT32, bias_buffer_id));

  const int output_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
      TensorType_FLOAT32));

  const int output_shape_tensor_id = tensors.size();
  const std::array<int32_t, 1> output_shape_shape{{4}};
  tensors.emplace_back(
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape_shape.data(),
                                                 output_shape_shape.size()),
                   TensorType_INT32, output_shape_buffer_id));

  /***************************** Define operators *****************************/
  std::vector<flatbuffers::Offset<tflite::Operator>> operators;

  std::vector<int32_t> op_inputs{{output_shape_tensor_id, filter_tensor_id,
                                  input_tensor_id, bias_tensor_id}};
  const std::array<int32_t, 1> op_outputs{{output_tensor_id}};
  const flatbuffers::Offset<TransposeConvOptions> transpose_conv_options =
      CreateTransposeConvOptions(builder, Padding(), StrideWidth(),
                                 StrideHeight());
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_TransposeConvOptions, transpose_conv_options.Union()));

  /****************************** Define subgraph *****************************/
  const std::array<int32_t, 1> subgraph_inputs{{input_tensor_id}};
  const std::array<int32_t, 1> subgraph_outputs{{output_tensor_id}};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  const flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Dynamically Quantized Transpose Conv2D model");

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

std::vector<char>
DynamicallyQuantizedTransposeConvTester::CreateDequantizeTfLiteModel(
    const std::vector<int8_t>& filter_data, const std::vector<float>& bias_data,
    const std::vector<float>& kernel_scale) const {
  /*************************** Define operator codes **************************/
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_TRANSPOSE_CONV)}};
  const int dequantize_operator_code = operator_codes.size();
  operator_codes.emplace_back(
      CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));

  /****************************** Define buffers ******************************/
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({}))}};

  int filter_buffer_id = 0;
  const int quantized_filter_buffer_id = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(filter_data.data()),
                           sizeof(int8_t) * filter_data.size())));

  int bias_buffer_id = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(bias_data.data()),
                           sizeof(float) * bias_data.size())));
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), OutputChannels()}};
  const int output_shape_buffer_id = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder, builder.CreateVector(
                   reinterpret_cast<const uint8_t*>(output_shape.data()),
                   sizeof(int32_t) * output_shape.size())));

  /****************************** Define tensors ******************************/
  const std::vector<int32_t> filter_shape = {OutputChannels(), KernelHeight(),
                                             KernelWidth(), InputChannels()};
  const std::vector<int32_t> bias_shape = {OutputChannels()};
  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), InputChannels()}};

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  const int quantized_filter_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
      /*type=*/TensorType_INT8,
      /*buffer=*/quantized_filter_buffer_id,
      /*name=*/0,
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>(kernel_scale),
          builder.CreateVector<int64_t>(
              std::vector<int64_t>(OutputChannels(), 0)))));

  const int input_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
      TensorType_FLOAT32));

  const int filter_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
      TensorType_FLOAT32,
      /*buffer=*/filter_buffer_id));

  const int bias_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(bias_shape.data(), bias_shape.size()),
      TensorType_FLOAT32, bias_buffer_id));

  const int output_tensor_id = tensors.size();
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
      TensorType_FLOAT32));

  const int output_shape_tensor_id = tensors.size();
  const std::array<int32_t, 1> output_shape_shape{{4}};
  tensors.emplace_back(
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape_shape.data(),
                                                 output_shape_shape.size()),
                   TensorType_INT32, output_shape_buffer_id));

  /***************************** Define operators *****************************/
  std::vector<flatbuffers::Offset<tflite::Operator>> operators;

  const std::array<int32_t, 1> dequantize_filter_inputs{
      {quantized_filter_tensor_id}};
  const std::array<int32_t, 1> dequantize_filter_outputs{{filter_tensor_id}};
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/dequantize_operator_code,
      builder.CreateVector<int32_t>(dequantize_filter_inputs.data(),
                                    dequantize_filter_inputs.size()),
      builder.CreateVector<int32_t>(dequantize_filter_outputs.data(),
                                    dequantize_filter_outputs.size())));

  std::vector<int32_t> op_inputs{
      {output_shape_tensor_id, filter_tensor_id, input_tensor_id}};
  op_inputs.push_back(bias_tensor_id);
  const std::array<int32_t, 1> op_outputs{{output_tensor_id}};
  const flatbuffers::Offset<TransposeConvOptions> transpose_conv_options =
      CreateTransposeConvOptions(builder, Padding(), StrideWidth(),
                                 StrideHeight());
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_TransposeConvOptions, transpose_conv_options.Union()));

  /****************************** Define subgraph *****************************/
  const std::array<int32_t, 1> subgraph_inputs{{input_tensor_id}};
  const std::array<int32_t, 1> subgraph_outputs{{output_tensor_id}};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  const flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("TransposeConv model");

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
