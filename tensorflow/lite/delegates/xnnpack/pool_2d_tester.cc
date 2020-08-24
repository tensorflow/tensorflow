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

#include "tensorflow/lite/delegates/xnnpack/pool_2d_tester.h"

#include <array>
#include <cstdint>
#include <functional>
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

void Pool2DTester::Test(tflite::BuiltinOperator pool_op,
                        TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto range_rng = std::bind(
      std::uniform_real_distribution<float>(-25.0f, 25.0f), std::ref(rng));

  std::vector<char> buffer = CreateTfLiteModel(pool_op);
  const tflite::Model* model = tflite::GetModel(buffer.data());

  std::unique_ptr<tflite::Interpreter> delegate_interpreter;
  ASSERT_EQ(tflite::InterpreterBuilder(
                model, tflite::ops::builtin::BuiltinOpResolver())(
                &delegate_interpreter),
            kTfLiteOk);
  std::unique_ptr<tflite::Interpreter> default_interpreter;
  ASSERT_EQ(tflite::InterpreterBuilder(
                model, tflite::ops::builtin::BuiltinOpResolver())(
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
  for (int32_t i = 0; i < BatchSize(); i++) {
    for (int32_t c = 0; c < Channels(); c++) {
      // Use the same range of all-positive or all-negative values to generate
      // all pixels within the same batch index & channel, but different ranges
      // for different channels or batches. This ensures that no catastrophic
      // cancellation occur, but test covers both positive and negative inputs.
      const float range = range_rng();
      auto value_rng =
          std::bind(std::uniform_real_distribution<float>(
                        std::min(range, 0.0f), std::max(range, 0.0f)),
                    std::ref(rng));
      for (int32_t y = 0; y < InputHeight(); y++) {
        for (int32_t x = 0; x < InputWidth(); x++) {
          const int32_t index =
              ((i * InputHeight() + y) * InputWidth() + x) * Channels() + c;
          default_input_data[index] = value_rng();
        }
      }
    }
  }

  float* xnnpack_input_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->inputs()[0]);
  std::copy(default_input_data,
            default_input_data +
                BatchSize() * InputHeight() * InputWidth() * Channels(),
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data = default_interpreter->typed_tensor<float>(
      default_interpreter->outputs()[0]);
  float* xnnpack_output_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->outputs()[0]);

  for (int32_t i = 0; i < BatchSize(); i++) {
    for (int32_t y = 0; y < OutputHeight(); y++) {
      for (int32_t x = 0; x < OutputWidth(); x++) {
        for (int32_t c = 0; c < Channels(); c++) {
          const int32_t index =
              ((i * OutputHeight() + y) * OutputWidth() + x) * Channels() + c;
          if (pool_op == BuiltinOperator_MAX_POOL_2D) {
            // MaxPooling results must be exact
            ASSERT_EQ(default_output_data[index], xnnpack_output_data[index])
                << "batch " << i << " / " << BatchSize() << ", y position " << y
                << " / " << OutputHeight() << ", x position " << x << " / "
                << OutputWidth() << ", channel " << c << " / " << Channels();
          } else {
            ASSERT_NEAR(default_output_data[index], xnnpack_output_data[index],
                        std::abs(default_output_data[index]) * 3.0e-6f)
                << "batch " << i << " / " << BatchSize() << ", y position " << y
                << " / " << OutputHeight() << ", x position " << x << " / "
                << OutputWidth() << ", channel " << c << " / " << Channels();
          }
        }
      }
    }
  }
}

std::vector<char> Pool2DTester::CreateTfLiteModel(
    tflite::BuiltinOperator pool_op) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, pool_op, 0);

  flatbuffers::Offset<tflite::Pool2DOptions> pool_2d_options =
      CreatePool2DOptions(builder, Padding(), StrideWidth(), StrideHeight(),
                          PoolingWidth(), PoolingHeight(), Activation());

  const flatbuffers::Offset<tflite::Buffer> null_buffer =
      tflite::CreateBuffer(builder, builder.CreateVector({}));

  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), Channels()}};
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), Channels()}};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
      tflite::CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          tflite::TensorType_FLOAT32),
      tflite::CreateTensor(builder,
                           builder.CreateVector<int32_t>(output_shape.data(),
                                                         output_shape.size()),
                           tflite::TensorType_FLOAT32),
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};

  flatbuffers::Offset<tflite::Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_Pool2DOptions, pool_2d_options.Union());

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Pool2D model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(&null_buffer, 1));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
