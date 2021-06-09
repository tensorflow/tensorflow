/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/quantized_conv_2d_tester.h"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void QuantizedConv2DTester::Test(TfLiteDelegate* delegate) const {
  std::vector<char> buffer = CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                 std::numeric_limits<int8_t>::min(),
                                 std::numeric_limits<int8_t>::max()),
                             std::ref(rng));
  int8_t* default_input_data = default_interpreter->typed_tensor<int8_t>(
      default_interpreter->inputs()[0]);
  std::generate(default_input_data,
                default_input_data + BatchSize() * InputHeight() *
                                         InputWidth() * InputChannels(),
                input_rng);

  int8_t* delegate_input_data = delegate_interpreter->typed_tensor<int8_t>(
      delegate_interpreter->inputs()[0]);
  std::copy(default_input_data,
            default_input_data +
                BatchSize() * InputHeight() * InputWidth() * InputChannels(),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  int8_t* default_output_data = default_interpreter->typed_tensor<int8_t>(
      default_interpreter->outputs()[0]);
  int8_t* delegate_output_data = delegate_interpreter->typed_tensor<int8_t>(
      delegate_interpreter->outputs()[0]);

  for (int32_t i = 0; i < BatchSize(); i++) {
    for (int32_t y = 0; y < OutputHeight(); y++) {
      for (int32_t x = 0; x < OutputWidth(); x++) {
        for (int32_t c = 0; c < OutputChannels(); c++) {
          const int32_t index = ((i * OutputHeight() + y) * OutputWidth() + x) *
                                    OutputChannels() +
                                c;
          ASSERT_LE(std::abs(static_cast<int32_t>(default_output_data[index]) -
                             static_cast<int32_t>(delegate_output_data[index])),
                    1)
              << "default " << static_cast<int32_t>(default_output_data[index])
              << ", delegate "
              << static_cast<int32_t>(delegate_output_data[index]) << ", batch "
              << i << " / " << BatchSize() << ", y position " << y << " / "
              << OutputHeight() << ", x position " << x << " / "
              << OutputWidth() << ", channel " << c << " / "
              << OutputChannels();
        }
      }
    }
  }
}

std::vector<char> QuantizedConv2DTester::CreateTfLiteModel() const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto filter_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                  -std::numeric_limits<int8_t>::max(),
                                  std::numeric_limits<int8_t>::max()),
                              std::ref(rng));
  auto bias_rng = std::bind(
      std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  const std::array<flatbuffers::Offset<OperatorCode>, 1> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_CONV_2D)}};

  std::vector<int8_t> filter_data(OutputChannels() * KernelHeight() *
                                  KernelWidth() * InputChannels());
  std::generate(filter_data.begin(), filter_data.end(), std::ref(filter_rng));
  std::vector<int32_t> bias_data(OutputChannels());
  std::generate(bias_data.begin(), bias_data.end(), std::ref(bias_rng));

  const std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(filter_data.data()),
                       sizeof(int8_t) * filter_data.size())),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(bias_data.data()),
                       sizeof(int32_t) * bias_data.size())),
  }};

  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), InputChannels()}};
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), OutputChannels()}};
  const std::array<int32_t, 4> filter_shape{
      {OutputChannels(), KernelHeight(), KernelWidth(), InputChannels()}};
  const std::array<int32_t, 1> bias_shape{{OutputChannels()}};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 4> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          TensorType_INT8, /*buffer=*/0, /*name=*/0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({InputScale()}),
              builder.CreateVector<int64_t>({InputZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(filter_shape.data(),
                                                 filter_shape.size()),
                   TensorType_INT8, /*buffer=*/1, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({KernelScale()}),
                       builder.CreateVector<int64_t>({0}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(bias_shape.data(), bias_shape.size()),
          TensorType_INT32, /*buffer=*/2, /*name=*/0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({InputScale() * KernelScale()}),
              builder.CreateVector<int64_t>({0}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape.data(),
                                                 output_shape.size()),
                   TensorType_INT8, /*buffer=*/0, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({OutputScale()}),
                       builder.CreateVector<int64_t>({OutputZeroPoint()}))),
  }};

  const std::array<int32_t, 3> op_inputs{{0, 1, 2}};
  const std::array<int32_t, 1> op_outputs{{3}};
  const flatbuffers::Offset<Conv2DOptions> conv2d_options =
      CreateConv2DOptions(builder, Padding(), StrideWidth(), StrideHeight(),
                          Activation(), DilationWidth(), DilationHeight());

  const std::array<flatbuffers::Offset<tflite::Operator>, 1> operators{
      {CreateOperator(
          builder, /*opcode_index=*/0,
          builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
          builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
          BuiltinOptions_Conv2DOptions, conv2d_options.Union())}};

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{3}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Quantized Conv2D model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
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
