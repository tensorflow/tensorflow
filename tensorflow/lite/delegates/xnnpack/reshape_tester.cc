/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/reshape_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void ReshapeTester::Test(TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  ASSERT_EQ(InputSize(), OutputSize());

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

  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate(default_input_data, default_input_data + InputSize(),
                std::ref(f32rng));

  float* delegate_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy(default_input_data, default_input_data + InputSize(),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  float* delegate_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  for (size_t i = 0; i < OutputSize(); i++) {
    ASSERT_EQ(delegate_output_data[i], default_output_data[i]);
  }
}

std::vector<char> ReshapeTester::CreateTfLiteModel() const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_RESHAPE, 0);

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};
  if (OutputShapeAsInput()) {
    buffers.emplace_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(OutputShape().data()),
                     OutputShape().size() * sizeof(int32_t))));
  }

  std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   TensorType_FLOAT32),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(OutputShape().data(),
                                                 OutputShape().size()),
                   TensorType_FLOAT32),
  }};

  if (OutputShapeAsInput()) {
    const std::array<int32_t, 1> reshape_shape{
        {static_cast<int32_t>(InputShape().size())}};
    tensors.insert(tensors.begin() + 1,
                   CreateTensor(builder,
                                builder.CreateVector<int32_t>(
                                    reshape_shape.data(), reshape_shape.size()),
                                TensorType_INT32, /*buffer=*/1));
  }

  std::vector<int32_t> op_inputs({0});
  if (OutputShapeAsInput()) {
    op_inputs.push_back(1);
  }
  const std::array<int32_t, 1> op_outputs{{OutputShapeAsInput() ? 2 : 1}};

  BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;
  if (!OutputShapeAsInput()) {
    builtin_options_type = tflite::BuiltinOptions_ReshapeOptions;
    builtin_options =
        CreateReshapeOptions(
            builder, builder.CreateVector<int32_t>(OutputShape().data(),
                                                   OutputShape().size()))
            .Union();
  }

  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      builtin_options_type, builtin_options);

  const std::array<int32_t, 1> subgraph_inputs{{op_inputs.front()}};
  const std::array<int32_t, 1> subgraph_outputs{{op_outputs.front()}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Reshape model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t ReshapeTester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
