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

#include "tensorflow/lite/delegates/xnnpack/slice_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

template <typename T>
std::function<T(std::mt19937&)> GetDist() {
  return std::uniform_int_distribution<int32_t>(std::numeric_limits<T>::min(),
                                                std::numeric_limits<T>::max());
}

template <>
std::function<float(std::mt19937&)> GetDist() {
  return std::uniform_real_distribution<float>();
}

template <typename T>
void SliceTester::Test(Interpreter* default_interpreter,
                       Interpreter* delegate_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_distribution = GetDist<T>();
  auto input_rng = std::bind(input_distribution, std::ref(rng));
  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::generate(default_input_data,
                default_input_data + ComputeSize(InputShape()),
                std::ref(input_rng));

  T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy(default_input_data, default_input_data + ComputeSize(InputShape()),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    EXPECT_EQ(default_output_data[i], delegate_output_data[i]);
  }
}

void SliceTester::Test(TensorType tensor_type, TfLiteDelegate* delegate) const {
  ASSERT_EQ(InputShape().size(), Offsets().size());
  ASSERT_EQ(InputShape().size(), Sizes().size());
  for (size_t i = 0; i < InputShape().size(); i++) {
    ASSERT_GE(Offsets()[i], 0);
    ASSERT_LT(Offsets()[i], InputShape()[i]);
    if (Sizes()[i] < 0) {
      ASSERT_EQ(Sizes()[i], -1);
      ASSERT_EQ(InputShape()[i] - Offsets()[i], OutputShape()[i]);
    } else {
      ASSERT_GT(Sizes()[i], 0);
      ASSERT_LE(Sizes()[i], InputShape()[i]);
      ASSERT_EQ(Sizes()[i], OutputShape()[i]);
    }
    ASSERT_LE(Offsets()[i] + Sizes()[i], InputShape()[i]);
  }

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

std::vector<char> SliceTester::CreateTfLiteModel(TensorType tensor_type) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_SLICE);

  const std::array<flatbuffers::Offset<Buffer>, 3> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder, builder.CreateVector(
                                reinterpret_cast<const uint8_t*>(OffsetsData()),
                                OffsetsSizeInBytes())),
      CreateBuffer(builder, builder.CreateVector(
                                reinterpret_cast<const uint8_t*>(SizesData()),
                                SizesSizeInBytes())),
  }};

  flatbuffers::Offset<QuantizationParameters> quantization_params =
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>({/*scale=*/1.0f}),
          builder.CreateVector<int64_t>({/*zero_point=*/0}));

  const int32_t num_dims = Offsets().size();
  TensorType offsets_and_sizes_tensor_type =
      UseInt64OffsetsAndSize() ? TensorType_INT64 : TensorType_INT32;

  const std::array<flatbuffers::Offset<Tensor>, 4> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   tensor_type, /*buffer=*/0, /*name=*/0, quantization_params),
      CreateTensor(builder, builder.CreateVector<int32_t>({num_dims}),
                   offsets_and_sizes_tensor_type, /*buffer=*/1),
      CreateTensor(builder, builder.CreateVector<int32_t>({num_dims}),
                   offsets_and_sizes_tensor_type, /*buffer=*/2),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(OutputShape().data(),
                                                 OutputShape().size()),
                   tensor_type, /*buffer=*/0, /*name=*/0, quantization_params),
  }};

  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0, builder.CreateVector<int32_t>({0, 1, 2}),
      builder.CreateVector<int32_t>({3}));

  const flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>({0}), builder.CreateVector<int32_t>({3}),
      builder.CreateVector({op}));

  const flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Slice model");

  const flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t ComputeSize(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

std::vector<int32_t> RandomOffsets(std::mt19937& rng,
                                   const std::vector<int32_t>& dims) {
  std::vector<int32_t> offsets(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    offsets[i] = std::uniform_int_distribution<int32_t>(0, dims[i] - 1)(rng);
  }
  return offsets;
}

std::vector<int32_t> RandomSizes(std::mt19937& rng,
                                 const std::vector<int32_t>& dims,
                                 const std::vector<int32_t>& offsets) {
  std::vector<int32_t> sizes(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    // Allow -1 as a size (which means select everything).
    std::vector<int32_t> valid_sizes(dims[i] - offsets[i] + 1);
    std::iota(valid_sizes.begin(), valid_sizes.end(), 1);
    valid_sizes.back() = -1;
    sizes[i] = valid_sizes[std::uniform_int_distribution<int32_t>(
        0, valid_sizes.size() - 1)(rng)];
  }
  return sizes;
}

}  // namespace xnnpack
}  // namespace tflite
