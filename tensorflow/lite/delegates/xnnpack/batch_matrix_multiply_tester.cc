/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/batch_matrix_multiply_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
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

std::vector<int32_t> BatchMatrixMultiplyTester::OutputShape() const {
  std::vector<int32_t> output_shape = Input1Shape();
  const size_t output_dimensions = output_shape.size();
  output_shape[output_dimensions - 1] =
      AdjY() ? Input2Shape()[Input2Shape().size() - 2]
             : Input2Shape()[Input2Shape().size() - 1];
  return output_shape;
}

void BatchMatrixMultiplyTester::Test(TfLiteDelegate* delegate) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

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

  ASSERT_EQ(delegate_interpreter->inputs().size(), 2);
  ASSERT_EQ(default_interpreter->inputs().size(), 2);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  if (weights_cache_ != nullptr) {
    TfLiteXNNPackDelegateWeightsCacheFinalizeHard(weights_cache_);
  }

  float* default_input1_data =
      default_interpreter->typed_input_tensor<float>(0);
  float* default_input2_data =
      default_interpreter->typed_input_tensor<float>(1);
  std::generate_n(default_input1_data, Input1Size(), std::ref(input_rng));
  std::generate_n(default_input2_data, Input2Size(), std::ref(input_rng));

  float* delegate_input1_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  float* delegate_input2_data =
      delegate_interpreter->typed_input_tensor<float>(1);
  std::copy_n(default_input1_data, Input1Size(), delegate_input1_data);
  std::copy_n(default_input2_data, Input2Size(), delegate_input2_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  float* delegate_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  const int32_t output_size = ComputeSize(OutputShape());
  for (size_t i = 0; i < output_size; i++) {
    ASSERT_NEAR(default_output_data[i], delegate_output_data[i],
                std::numeric_limits<float>::epsilon() *
                    std::max(std::abs(default_output_data[i]) * 20.0f, 1.0f));
  }
}

std::vector<char> BatchMatrixMultiplyTester::CreateTfLiteModel() const {
  /*************************** Define operator codes **************************/
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_BATCH_MATMUL)}};

  /****************************** Define buffers ******************************/
  std::vector<flatbuffers::Offset<Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({})),
       CreateBuffer(builder, builder.CreateVector({}))}};

  /****************************** Define tensors ******************************/
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(Input1Shape().data(), Input1Shape().size()),
      TensorType_FLOAT32, /*buffer=*/0));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(Input2Shape().data(), Input2Shape().size()),
      TensorType_FLOAT32, /*buffer=*/0));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(OutputShape().data(), OutputShape().size()),
      TensorType_FLOAT32));

  /***************************** Define operators *****************************/

  std::vector<int32_t> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  const flatbuffers::Offset<BatchMatMulOptions> batch_matmul_options =
      CreateBatchMatMulOptions(builder, AdjX(), AdjY());
  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_BatchMatMulOptions, batch_matmul_options.Union());

  /****************************** Define subgraph *****************************/
  const std::array<int32_t, 2> subgraph_inputs{{0, 1}};
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Batch Matrix Multiply model");

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t BatchMatrixMultiplyTester::ComputeSize(
    const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
