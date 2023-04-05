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

#include "tensorflow/lite/delegates/xnnpack/split_tester.h"

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
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

template <class T>
void SplitTester::Test(Interpreter *delegate_interpreter,
                       Interpreter *default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int32_t> input_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  T *default_input_data = default_interpreter->typed_input_tensor<T>(1);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));

  T *xnnpack_input_data = delegate_interpreter->typed_input_tensor<T>(1);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T *default_output1_data = default_interpreter->typed_output_tensor<T>(0);
  T *xnnpack_output1_data = delegate_interpreter->typed_output_tensor<T>(0);
  T *default_output2_data = default_interpreter->typed_output_tensor<T>(1);
  T *xnnpack_output2_data = delegate_interpreter->typed_output_tensor<T>(1);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(static_cast<int32_t>(default_output1_data[i]),
              static_cast<int32_t>(xnnpack_output1_data[i]));
    ASSERT_EQ(static_cast<int32_t>(default_output2_data[i]),
              static_cast<int32_t>(xnnpack_output2_data[i]));
  }
}

template <>
void SplitTester::Test<float>(Interpreter *delegate_interpreter,
                              Interpreter *default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input_distribution(-25.0f, 25.0f);
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  float *default_input_data = default_interpreter->typed_input_tensor<float>(1);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));

  float *xnnpack_input_data =
      delegate_interpreter->typed_input_tensor<float>(1);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float *default_output1_data =
      default_interpreter->typed_output_tensor<float>(0);
  float *xnnpack_output1_data =
      delegate_interpreter->typed_output_tensor<float>(0);
  float *default_output2_data =
      default_interpreter->typed_output_tensor<float>(0);
  float *xnnpack_output2_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(default_output1_data[i], xnnpack_output1_data[i]);
    ASSERT_EQ(default_output2_data[i], xnnpack_output2_data[i]);
  }
}

void SplitTester::Test(TensorType tensor_type, TfLiteDelegate *delegate) const {
  std::vector<char> buffer = CreateTfLiteModel(tensor_type);
  const Model *model = GetModel(buffer.data());

  int32_t axis = SplitDimension();
  axis += axis < 0 ? InputShape().size() : 0;
  ASSERT_EQ(0, InputShape()[axis] % NumSplits());

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
  ASSERT_EQ(delegate_interpreter->outputs().size(), NumSplits());
  ASSERT_EQ(default_interpreter->outputs().size(), NumSplits());

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  switch (tensor_type) {
    case TensorType_FLOAT32:
      Test<float>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case TensorType_INT8:
      Test<int8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case TensorType_UINT8:
      Test<uint8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    default:
      GTEST_FAIL();
  }
}

std::vector<char> SplitTester::CreateTfLiteModel(TensorType tensor_type) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_SPLIT, 0);

  std::array<int32_t, 1> split_dim = {SplitDimension()};
  std::vector<flatbuffers::Offset<Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({})),
       CreateBuffer(builder,
                    builder.CreateVector(
                        reinterpret_cast<const uint8_t *>(split_dim.data()),
                        split_dim.size() * sizeof(int32_t)))}};
  std::array<int32_t, 0> split_dim_shape = {};

  flatbuffers::Offset<QuantizationParameters> quantization_params =
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>({/*scale=*/1.0f}),
          builder.CreateVector<int64_t>({/*zero_point=*/0}));

  std::vector<flatbuffers::Offset<Tensor>> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(split_dim_shape.data(),
                                                 split_dim_shape.size()),
                   TensorType_INT32, /*buffer=*/1, /*name=*/0,
                   quantization_params),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   tensor_type,
                   /*buffer=*/0, /*name=*/0, quantization_params),
  }};

  for (int i = 0; i < NumSplits(); i++) {
    tensors.push_back(
        CreateTensor(builder,
                     builder.CreateVector<int32_t>(OutputShape().data(),
                                                   OutputShape().size()),
                     tensor_type,
                     /*buffer=*/0, /*name=*/0, quantization_params));
  }

  const std::array<int32_t, 2> op_inputs{0, 1};
  std::vector<int32_t> op_outputs;
  op_outputs.reserve(NumSplits());
  for (int i = 0; i < NumSplits(); i++) {
    op_outputs.push_back(op_inputs.size() + i);
  }
  EXPECT_EQ(op_outputs.size(), NumSplits());

  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_SplitOptions,
      CreateSplitOptions(builder, NumSplits()).Union());

  const std::array<int32_t, 2> subgraph_inputs = op_inputs;
  const std::vector<int32_t> subgraph_outputs = op_outputs;
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Split model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t SplitTester::ComputeSize(const std::vector<int32_t> &shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
