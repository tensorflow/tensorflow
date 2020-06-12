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

#include "tensorflow/lite/delegates/xnnpack/reduce_tester.h"

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
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void ReduceTester::Test(tflite::BuiltinOperator reduce_op,
                        TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng = std::bind(
      std::uniform_real_distribution<float>(-15.0f, 15.0f), std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel(reduce_op);
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

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  float* default_input_data = default_interpreter->typed_tensor<float>(
      default_interpreter->inputs()[0]);
  std::generate(default_input_data, default_input_data + InputSize(),
                std::ref(input_rng));

  float* delegate_input_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->inputs()[0]);
  std::copy(default_input_data, default_input_data + InputSize(),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data = default_interpreter->typed_tensor<float>(
      default_interpreter->outputs()[0]);
  float* delegate_output_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->outputs()[0]);

  const int32_t output_size = OutputSize();
  for (size_t i = 0; i < output_size; i++) {
    ASSERT_NEAR(
        default_output_data[i], delegate_output_data[i],
        std::numeric_limits<float>::epsilon() *
            std::max(std::abs(default_output_data[i]) * RelativeTolerance(),
                     1.0f));
  }
}

std::vector<char> ReduceTester::CreateTfLiteModel(
    tflite::BuiltinOperator reduce_op) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, reduce_op);

  const std::array<flatbuffers::Offset<Buffer>, 2> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder, builder.CreateVector(
                                reinterpret_cast<const uint8_t*>(Axes().data()),
                                sizeof(int32_t) * Axes().size())),
  }};

  const std::vector<int32_t> output_shape = OutputShape();
  const std::array<int32_t, 1> axes_shape{
      {static_cast<int32_t>(Axes().size())}};
  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   TensorType_FLOAT32),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(axes_shape.data(), axes_shape.size()),
          TensorType_INT32, /*buffer=*/1),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape.data(),
                                                 output_shape.size()),
                   TensorType_FLOAT32),
  }};

  const flatbuffers::Offset<ReducerOptions> reducer_options =
      CreateReducerOptions(builder, KeepDims());

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_ReducerOptions, reducer_options.Union());

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Reduce model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t ReduceTester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
