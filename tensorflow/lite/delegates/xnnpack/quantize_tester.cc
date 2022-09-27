/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/quantize_tester.h"

#include <algorithm>
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

template <class T>
void QuantizeTester::PopulateInput(Interpreter* delegate_interpreter,
                                   Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int> input_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::generate(default_input_data, default_input_data + ComputeSize(Shape()),
                std::ref(input_rng));

  T* xnnpack_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy(default_input_data, default_input_data + ComputeSize(Shape()),
            xnnpack_input_data);
}

template <>
void QuantizeTester::PopulateInput<float>(
    Interpreter* delegate_interpreter, Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input_distribution(-1.0f, 1.0f);
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate(default_input_data, default_input_data + ComputeSize(Shape()),
                std::ref(input_rng));

  float* xnnpack_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy(default_input_data, default_input_data + ComputeSize(Shape()),
            xnnpack_input_data);
}

template <class T>
void QuantizeTester::InvokeAndCheckOutput(
    Interpreter* delegate_interpreter, Interpreter* default_interpreter) const {
  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (size_t i = 0; i < ComputeSize(Shape()); i++) {
    ASSERT_LE(std::abs(static_cast<int32_t>(default_output_data[i]) -
                       static_cast<int32_t>(delegate_output_data[i])),
              1)
        << "default " << static_cast<int32_t>(default_output_data[i])
        << ", delegate " << static_cast<int32_t>(delegate_output_data[i])
        << " at index " << i << " / " << ComputeSize(Shape());
  }
}

void QuantizeTester::Test(TensorType input_type, TensorType output_type,
                          TfLiteDelegate* delegate) const {
  std::vector<char> buffer = CreateTfLiteModel(input_type, output_type);
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

  switch (input_type) {
    case TensorType_FLOAT32:
      PopulateInput<float>(delegate_interpreter.get(),
                           default_interpreter.get());
      break;
    case TensorType_INT8:
      PopulateInput<int8_t>(delegate_interpreter.get(),
                            default_interpreter.get());
      break;
    case TensorType_UINT8:
      PopulateInput<uint8_t>(delegate_interpreter.get(),
                             default_interpreter.get());
      break;
    default:
      GTEST_FAIL() << "unsupported input type "
                   << EnumNameTensorType(input_type);
  }

  switch (output_type) {
    case TensorType_INT8:
      InvokeAndCheckOutput<int8_t>(delegate_interpreter.get(),
                                   default_interpreter.get());
      break;
    case TensorType_UINT8:
      InvokeAndCheckOutput<uint8_t>(delegate_interpreter.get(),
                                    default_interpreter.get());
      break;
    default:
      GTEST_FAIL() << "unsupported output type "
                   << EnumNameTensorType(output_type);
  }
}

std::vector<char> QuantizeTester::CreateTfLiteModel(
    TensorType input_type, TensorType output_type) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_QUANTIZE);

  const std::array<flatbuffers::Offset<Buffer>, 1> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  flatbuffers::Offset<QuantizationParameters> input_quantization = 0;
  if (input_type != TensorType_FLOAT32) {
    input_quantization = CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        builder.CreateVector<float>({InputScale()}),
        builder.CreateVector<int64_t>({InputZeroPoint()}));
  }

  const std::array<flatbuffers::Offset<Tensor>, 2> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          input_type,
          /*buffer=*/0, /*name=*/0, input_quantization),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(Shape().data(), Shape().size()),
          output_type,
          /*buffer=*/0, /*name=*/0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({OutputScale()}),
              builder.CreateVector<int64_t>({OutputZeroPoint()}))),
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
      builder.CreateString("Quantize operator model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t QuantizeTester::ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
