/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/transpose_tester.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

template <class T>
void TransposeTester::Test(TensorType tensor_type,
                           Interpreter* delegate_interpreter,
                           Interpreter* default_interpreter) const {
  int32_t count = std::accumulate(input_shape().cbegin(), input_shape().cend(),
                                  1, std::multiplies<int32_t>());
  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::iota(default_input_data, default_input_data + count, 0);

  T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy(default_input_data, default_input_data + count,
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (int32_t i = 0; i < count; i++) {
    ASSERT_EQ(default_output_data[i], delegate_output_data[i]);
  }
}

void TransposeTester::Test(TensorType tensor_type,
                           TfLiteDelegate* delegate) const {
  const std::vector<char> buffer = CreateTfLiteModel(tensor_type);
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

  ASSERT_EQ(default_interpreter->inputs().size(), 1);
  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  switch (tensor_type) {
    case TensorType_FLOAT32:
      Test<float>(TensorType_FLOAT32, delegate_interpreter.get(),
                  default_interpreter.get());
      break;
    case TensorType_INT8:
      Test<int8_t>(TensorType_INT8, delegate_interpreter.get(),
                   default_interpreter.get());
      break;
    case TensorType_UINT8:
      Test<uint8_t>(TensorType_UINT8, delegate_interpreter.get(),
                    default_interpreter.get());
      break;
    default:
      GTEST_FAIL();
  }
}

std::vector<char> TransposeTester::CreateTfLiteModel(
    TensorType tensor_type) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_TRANSPOSE, 0);

  std::vector<flatbuffers::Offset<Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({})),
       CreateBuffer(builder, builder.CreateVector(
                                 reinterpret_cast<const uint8_t*>(perm_.data()),
                                 perm_.size() * sizeof(int32_t)))}};

  std::vector<int32_t> output_shape(input_shape().size());
  for (int32_t i = 0; i < perm().size(); ++i) {
    output_shape[i] = input_shape_[perm_[i]];
  }

  flatbuffers::Offset<QuantizationParameters> quantization_params =
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>({/*scale=*/1.0f}),
          builder.CreateVector<int64_t>({/*zero_point=*/0}));

  const std::array<int32_t, 1> perm_shape{
      {static_cast<int32_t>(perm().size())}};
  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{
      {CreateTensor(builder,
                    builder.CreateVector<int32_t>(input_shape().data(),
                                                  input_shape().size()),
                    tensor_type, 0, 0, quantization_params),
       CreateTensor(
           builder,
           builder.CreateVector<int32_t>(perm_shape.data(), perm_shape.size()),
           TensorType_INT32, 1, 0, quantization_params),
       CreateTensor(builder,
                    builder.CreateVector<int32_t>(output_shape.data(),
                                                  output_shape.size()),
                    tensor_type, 0, 0, quantization_params)}};

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_TransposeOptions,
      CreateTransposeOptions(builder).Union());

  const std::array<int32_t, 1> subgraph_inputs{{0}};
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("Transpose model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
