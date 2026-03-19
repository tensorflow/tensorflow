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

#include "tensorflow/lite/delegates/xnnpack/strided_slice_tester.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

int32_t ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

template <typename T>
void StridedSliceTester::Test(Interpreter* default_interpreter,
                              Interpreter* delegate_interpreter) const {
  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::iota(default_input_data, default_input_data + ComputeSize(InputShape()),
            0);

  T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  const T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  const T* delegate_output_data =
      delegate_interpreter->typed_output_tensor<T>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    EXPECT_EQ(default_output_data[i], delegate_output_data[i]);
  }
}

void StridedSliceTester::Test(TensorType tensor_type,
                              TfLiteDelegate* delegate) const {
  ASSERT_EQ(InputShape().size(), Begins().size());
  ASSERT_EQ(InputShape().size(), Ends().size());
  ASSERT_EQ(InputShape().size(), Strides().size());
  for (size_t i = 0; i < InputShape().size(); i++) {
    ASSERT_GT(InputShape()[i], 0);
    ASSERT_LT(Begin(i), InputShape()[i]);
    ASSERT_LE(End(i), InputShape()[i]);
    ASSERT_EQ(End(i) - Begin(i), OutputShape()[i]);
    ASSERT_EQ(Strides()[i], 1);
  }

  std::vector<char> buffer = CreateTfLiteModel(tensor_type);
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

std::vector<char> StridedSliceTester::CreateTfLiteModel(
    TensorType tensor_type) const {
  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_STRIDED_SLICE);

  const std::array<flatbuffers::Offset<Buffer>, 4> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(Begins().data()),
                       sizeof(int32_t) * Begins().size())),
      CreateBuffer(builder, builder.CreateVector(
                                reinterpret_cast<const uint8_t*>(Ends().data()),
                                sizeof(int32_t) * Ends().size())),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(Strides().data()),
                       sizeof(int32_t) * Strides().size())),
  }};

  const flatbuffers::Offset<QuantizationParameters> quantization_params =
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>({/*scale=*/1.0f}),
          builder.CreateVector<int64_t>({/*zero_point=*/0}));

  const int32_t num_dims = Begins().size();
  const std::array<flatbuffers::Offset<Tensor>, 5> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   tensor_type, /*buffer=*/0, /*name=*/0, quantization_params),
      CreateTensor(builder, builder.CreateVector<int32_t>({num_dims}),
                   TensorType_INT32, /*buffer=*/1),
      CreateTensor(builder, builder.CreateVector<int32_t>({num_dims}),
                   TensorType_INT32, /*buffer=*/2),
      CreateTensor(builder, builder.CreateVector<int32_t>({num_dims}),
                   TensorType_INT32, /*buffer=*/3),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(OutputShape().data(),
                                                 OutputShape().size()),
                   tensor_type, /*buffer=*/0, /*name=*/0, quantization_params),
  }};

  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>({0, 1, 2, 3}),
      builder.CreateVector<int32_t>({4}),
      tflite::BuiltinOptions_StridedSliceOptions,
      CreateStridedSliceOptions(builder, BeginMask(), EndMask(), EllipsisMask(),
                                NewAxisMask(), ShrinkAxisMask())
          .Union());

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({4}),
      builder.CreateVector({op}));

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("StridedSlice model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
