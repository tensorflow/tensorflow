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

#include "tensorflow/lite/delegates/xnnpack/binary_elementwise_tester.h"

#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

std::vector<int32_t> BinaryElementwiseTester::OutputShape() const {
  std::vector<int32_t> output_shape;
  if (!input1_shape_.empty()) {
    output_shape.insert(
        output_shape.end(), input1_shape_.cbegin(),
        input1_shape_.cbegin() +
            std::max(input1_shape_.size(), input2_shape_.size()) -
            input2_shape_.size());
  }
  if (!input2_shape_.empty()) {
    output_shape.insert(
        output_shape.end(), input2_shape_.cbegin(),
        input2_shape_.cbegin() +
            std::max(input2_shape_.size(), input1_shape_.size()) -
            input1_shape_.size());
  }
  for (size_t i = std::min(input1_shape_.size(), input2_shape_.size()); i >= 1;
       i--) {
    output_shape.push_back(
        std::max(*(input1_shape_.cend() - i), *(input2_shape_.cend() - i)));
  }
  return output_shape;
}

void BinaryElementwiseTester::Test(tflite::BuiltinOperator binary_op,
                                   TfLiteDelegate* delegate) const {
  if (Input1Static()) {
    ASSERT_FALSE(Input2Static());
  }
  if (FP16Weights()) {
    ASSERT_TRUE(Input1Static() || Input2Static());
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input1_distribution(-25.0f, 25.0f);
  std::uniform_real_distribution<float> input2_distribution(-25.0f, 25.0f);
  switch (binary_op) {
    case BuiltinOperator_DIV:
      input1_distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
      input2_distribution = std::uniform_real_distribution<float>(0.1f, 1.0f);
      break;
    case BuiltinOperator_MUL:
      input1_distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
      input2_distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
      break;
    default:
      break;
  }
  auto input1_rng = std::bind(input1_distribution, std::ref(rng));
  auto input2_rng = std::bind(input2_distribution, std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel(binary_op);
  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(model, ::tflite::ops::builtin::BuiltinOpResolver())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(model, ::tflite::ops::builtin::BuiltinOpResolver())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  if (Input1Static() || Input2Static()) {
    ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
    ASSERT_EQ(default_interpreter->inputs().size(), 1);
  } else {
    ASSERT_EQ(delegate_interpreter->inputs().size(), 2);
    ASSERT_EQ(default_interpreter->inputs().size(), 2);
  }

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  if (!Input1Static()) {
    float* default_input1_data = default_interpreter->typed_tensor<float>(
        default_interpreter->inputs()[0]);
    std::generate(default_input1_data,
                  default_input1_data + ComputeSize(Input1Shape()),
                  std::ref(input1_rng));

    float* xnnpack_input1_data = delegate_interpreter->typed_tensor<float>(
        delegate_interpreter->inputs()[0]);
    std::copy(default_input1_data,
              default_input1_data + ComputeSize(Input1Shape()),
              xnnpack_input1_data);
  }

  if (!Input2Static()) {
    float* default_input2_data = default_interpreter->typed_tensor<float>(
        default_interpreter->inputs()[Input1Static() ? 0 : 1]);
    std::generate(default_input2_data,
                  default_input2_data + ComputeSize(Input2Shape()),
                  std::ref(input2_rng));

    float* xnnpack_input2_data = delegate_interpreter->typed_tensor<float>(
        delegate_interpreter->inputs()[Input1Static() ? 0 : 1]);
    std::copy(default_input2_data,
              default_input2_data + ComputeSize(Input2Shape()),
              xnnpack_input2_data);
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data = default_interpreter->typed_tensor<float>(
      default_interpreter->outputs()[0]);
  float* xnnpack_output_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->outputs()[0]);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_NEAR(default_output_data[i], xnnpack_output_data[i],
                std::numeric_limits<float>::epsilon() *
                    std::max(std::abs(default_output_data[i]) * 2.0f, 1.0f));
  }
}

std::vector<char> BinaryElementwiseTester::CreateTfLiteModel(
    tflite::BuiltinOperator binary_op) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input1_distribution(-25.0f, 25.0f);
  std::uniform_real_distribution<float> input2_distribution(-25.0f, 25.0f);
  switch (binary_op) {
    case BuiltinOperator_DIV:
      input1_distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
      input2_distribution = std::uniform_real_distribution<float>(0.1f, 1.0f);
      break;
    case BuiltinOperator_MUL:
      input1_distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
      input2_distribution = std::uniform_real_distribution<float>(-5.0f, 5.0f);
      break;
    default:
      break;
  }
  auto input1_rng = std::bind(input1_distribution, std::ref(rng));
  auto input2_rng = std::bind(input2_distribution, std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, binary_op)}};
  if (FP16Weights()) {
    operator_codes.emplace_back(
        CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));
  }

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  int32_t input1_buffer = 0;
  if (Input1Static()) {
    if (FP16Weights()) {
      std::vector<uint16_t> input1_data(ComputeSize(Input1Shape()));
      std::generate(input1_data.begin(), input1_data.end(),
                    std::bind(fp16_ieee_from_fp32_value, input1_rng));

      buffers.push_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(input1_data.data()),
                       sizeof(uint16_t) * input1_data.size())));
    } else {
      std::vector<float> input1_data(ComputeSize(Input1Shape()));
      std::generate(input1_data.begin(), input1_data.end(), input1_rng);

      input1_buffer = buffers.size();
      buffers.push_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(input1_data.data()),
                       sizeof(float) * input1_data.size())));
    }
  }

  int32_t input2_buffer = 0;
  if (Input2Static()) {
    if (FP16Weights()) {
      std::vector<uint16_t> input2_data(ComputeSize(Input2Shape()));
      std::generate(input2_data.begin(), input2_data.end(),
                    std::bind(fp16_ieee_from_fp32_value, input1_rng));

      buffers.push_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(input2_data.data()),
                       sizeof(uint16_t) * input2_data.size())));
    } else {
      std::vector<float> input2_data(ComputeSize(Input2Shape()));
      std::generate(input2_data.begin(), input2_data.end(), input2_rng);

      input2_buffer = buffers.size();
      buffers.push_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(input2_data.data()),
                       sizeof(float) * input2_data.size())));
    }
  }

  const std::vector<int32_t> output_shape = OutputShape();
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  std::vector<flatbuffers::Offset<Operator>> operators;
  if (FP16Weights() && Input1Static()) {
    tensors.emplace_back(
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(Input1Shape().data(),
                                                   Input1Shape().size()),
                     TensorType_FLOAT16, 1));
  }
  if (FP16Weights() && Input2Static()) {
    tensors.emplace_back(
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(Input2Shape().data(),
                                                   Input2Shape().size()),
                     TensorType_FLOAT16, 1));
  }
  if (FP16Weights()) {
    const std::array<int32_t, 1> dequantize_inputs{{0}};
    const std::array<int32_t, 1> dequantize_outputs{{Input1Static() ? 1 : 2}};
    operators.emplace_back(CreateOperator(
        builder, /*opcode_index=*/1,
        builder.CreateVector<int32_t>(dequantize_inputs.data(),
                                      dequantize_inputs.size()),
        builder.CreateVector<int32_t>(dequantize_outputs.data(),
                                      dequantize_outputs.size())));
  }
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(Input1Shape().data(), Input1Shape().size()),
      TensorType_FLOAT32, input1_buffer));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(Input2Shape().data(), Input2Shape().size()),
      TensorType_FLOAT32, input2_buffer));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
      TensorType_FLOAT32));

  tflite::BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;
  switch (binary_op) {
    case BuiltinOperator_ADD:
      builtin_options_type = BuiltinOptions_AddOptions;
      builtin_options = CreateAddOptions(builder, Activation()).Union();
      break;
    case BuiltinOperator_DIV:
      builtin_options_type = BuiltinOptions_DivOptions;
      builtin_options = CreateDivOptions(builder, Activation()).Union();
      break;
    case BuiltinOperator_MUL:
      builtin_options_type = BuiltinOptions_MulOptions;
      builtin_options = CreateMulOptions(builder, Activation()).Union();
      break;
    case BuiltinOperator_SUB:
      builtin_options_type = BuiltinOptions_SubOptions;
      builtin_options = CreateSubOptions(builder, Activation()).Union();
      break;
    default:
      EXPECT_EQ(Activation(), ActivationFunctionType_NONE);
  }

  const std::array<int32_t, 2> op_inputs{
      {static_cast<int>(tensors.size()) - 3,
       static_cast<int>(tensors.size()) - 2}};
  const std::array<int32_t, 1> op_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      builtin_options_type, builtin_options));

  std::vector<int32_t> subgraph_inputs;
  if (!Input1Static()) {
    subgraph_inputs.push_back(tensors.size() - 3);
  }
  if (!Input2Static()) {
    subgraph_inputs.push_back(tensors.size() - 2);
  }
  const std::array<int32_t, 1> subgraph_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Binary operator model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t BinaryElementwiseTester::ComputeSize(
    const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
