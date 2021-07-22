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

#include "tensorflow/lite/delegates/xnnpack/unary_elementwise_tester.h"

#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
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

void UnaryElementwiseTester::Test(tflite::BuiltinOperator unary_op,
                                  TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input_distribution(-15.0f, 15.0f);
  switch (unary_op) {
    case BuiltinOperator_SQRT:
      input_distribution = std::uniform_real_distribution<float>(0.0f, 10.0f);
      break;
    default:
      break;
  }
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel(unary_op);
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

  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate(default_input_data, default_input_data + Size(),
                std::ref(input_rng));

  float* delegate_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy(default_input_data, default_input_data + Size(),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  float* delegate_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  switch (unary_op) {
    case BuiltinOperator_ABS:
    case BuiltinOperator_CEIL:
    case BuiltinOperator_FLOOR:
    case BuiltinOperator_NEG:
    case BuiltinOperator_RELU:
    case BuiltinOperator_RELU_N1_TO_1:
    case BuiltinOperator_RELU6:
    case BuiltinOperator_ROUND:
    case BuiltinOperator_SQUARE:
    case BuiltinOperator_SQRT:
      for (size_t i = 0; i < Size(); i++) {
        ASSERT_EQ(default_output_data[i], delegate_output_data[i]);
      }
      break;
    default:
      for (size_t i = 0; i < Size(); i++) {
        ASSERT_NEAR(
            default_output_data[i], delegate_output_data[i],
            std::numeric_limits<float>::epsilon() *
                std::max(std::abs(default_output_data[i]) * RelativeTolerance(),
                         1.0f));
      }
      break;
  }
}

std::vector<char> UnaryElementwiseTester::CreateTfLiteModel(
    tflite::BuiltinOperator unary_op) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, unary_op);

  const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<flatbuffers::Offset<Tensor>, 2> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          TensorType_FLOAT32),
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{1}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Unary operator model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t UnaryElementwiseTester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
