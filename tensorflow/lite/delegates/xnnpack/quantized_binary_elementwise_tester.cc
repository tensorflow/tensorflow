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

#include "tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

std::vector<int32_t> QuantizedBinaryElementwiseTester::OutputShape() const {
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

template <class T>
void QuantizedBinaryElementwiseTester::Test(
    Interpreter* delegate_interpreter, Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int32_t> input1_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  std::uniform_int_distribution<int32_t> input2_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  auto input1_rng = std::bind(input1_distribution, std::ref(rng));
  auto input2_rng = std::bind(input2_distribution, std::ref(rng));
  if (!Input1Static()) {
    T* default_input1_data = default_interpreter->typed_input_tensor<T>(0);
    std::generate(default_input1_data,
                  default_input1_data + ComputeSize(Input1Shape()),
                  std::ref(input1_rng));

    T* xnnpack_input1_data = delegate_interpreter->typed_input_tensor<T>(0);
    std::copy(default_input1_data,
              default_input1_data + ComputeSize(Input1Shape()),
              xnnpack_input1_data);
  }

  if (!Input2Static()) {
    T* default_input2_data =
        default_interpreter->typed_input_tensor<T>(Input1Static() ? 0 : 1);
    std::generate(default_input2_data,
                  default_input2_data + ComputeSize(Input2Shape()),
                  std::ref(input2_rng));

    T* xnnpack_input2_data =
        delegate_interpreter->typed_input_tensor<T>(Input1Static() ? 0 : 1);
    std::copy(default_input2_data,
              default_input2_data + ComputeSize(Input2Shape()),
              xnnpack_input2_data);
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_LE(std::abs(static_cast<int32_t>(default_output_data[i]) -
                       static_cast<int32_t>(delegate_output_data[i])),
              1)
        << "default " << static_cast<int32_t>(default_output_data[i])
        << ", delegate " << static_cast<int32_t>(delegate_output_data[i])
        << " at index " << i << " / " << ComputeSize(OutputShape());
  }
}

void QuantizedBinaryElementwiseTester::Test(tflite::BuiltinOperator binary_op,
                                            TfLiteDelegate* delegate) const {
  if (Input1Static()) {
    ASSERT_FALSE(Input2Static());
  }

  std::vector<char> buffer = CreateTfLiteModel(binary_op);
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

  if (Unsigned()) {
    Test<uint8_t>(delegate_interpreter.get(), default_interpreter.get());
  } else {
    Test<int8_t>(delegate_interpreter.get(), default_interpreter.get());
  }
}

std::vector<char> QuantizedBinaryElementwiseTester::CreateTfLiteModel(
    tflite::BuiltinOperator binary_op) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int32_t> input1_distribution(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  std::uniform_int_distribution<int32_t> input2_distribution(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
  auto input1_rng = std::bind(input1_distribution, std::ref(rng));
  auto input2_rng = std::bind(input2_distribution, std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  const std::array<flatbuffers::Offset<OperatorCode>, 1> operator_codes{
      {CreateOperatorCode(builder, binary_op)}};

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  int32_t input1_buffer = 0;
  if (Input1Static()) {
    std::vector<int8_t> input1_data(ComputeSize(Input1Shape()));
    std::generate(input1_data.begin(), input1_data.end(), input1_rng);

    input1_buffer = buffers.size();
    buffers.push_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(input1_data.data()),
                     sizeof(int8_t) * input1_data.size())));
  }

  int32_t input2_buffer = 0;
  if (Input2Static()) {
    std::vector<int8_t> input2_data(ComputeSize(Input2Shape()));
    std::generate(input2_data.begin(), input2_data.end(), input2_rng);

    input2_buffer = buffers.size();
    buffers.push_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(input2_data.data()),
                     sizeof(int8_t) * input2_data.size())));
  }

  const std::vector<int32_t> output_shape = OutputShape();
  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input1Shape().data(),
                                                 Input1Shape().size()),
                   Unsigned() ? TensorType_UINT8 : TensorType_INT8,
                   input1_buffer, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({Input1Scale()}),
                       builder.CreateVector<int64_t>({Input1ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(Input2Shape().data(),
                                                 Input2Shape().size()),
                   Unsigned() ? TensorType_UINT8 : TensorType_INT8,
                   input2_buffer, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({Input2Scale()}),
                       builder.CreateVector<int64_t>({Input2ZeroPoint()}))),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(OutputShape().data(),
                                                 OutputShape().size()),
                   Unsigned() ? TensorType_UINT8 : TensorType_INT8,
                   /*buffer=*/0, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({OutputScale()}),
                       builder.CreateVector<int64_t>({OutputZeroPoint()}))),
  }};

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  const std::array<flatbuffers::Offset<Operator>, 1> operators{{CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_AddOptions,
      CreateAddOptions(builder, Activation()).Union())}};

  std::vector<int32_t> subgraph_inputs;
  if (!Input1Static()) {
    subgraph_inputs.push_back(0);
  }
  if (!Input2Static()) {
    subgraph_inputs.push_back(1);
  }
  const std::array<int32_t, 1> subgraph_outputs{{2}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Quantized binary operator model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t QuantizedBinaryElementwiseTester::ComputeSize(
    const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
