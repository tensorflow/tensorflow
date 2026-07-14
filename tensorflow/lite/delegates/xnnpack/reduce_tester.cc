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

#include "tensorflow/lite/delegates/xnnpack/reduce_tester.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

template <class T>
struct UniformDistribution {
  static std::uniform_int_distribution<int32_t> Get() {
    return std::uniform_int_distribution<int32_t>(
        std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  }
};

template <>
struct UniformDistribution<float> {
  static std::uniform_real_distribution<float> Get() { return {}; }
};

template <class T>
void ReduceTester::Test(Interpreter* delegate_interpreter,
                        Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng = std::bind(UniformDistribution<T>::Get(), std::ref(rng));

  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::generate_n(default_input_data, InputSize(), std::ref(input_rng));

  T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy_n(default_input_data, InputSize(), delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  const int32_t output_size = OutputSize();
  if constexpr (std::is_floating_point_v<T>) {
    for (size_t i = 0; i < output_size; i++) {
      ASSERT_NEAR(
          default_output_data[i], delegate_output_data[i],
          std::numeric_limits<float>::epsilon() *
              std::max(std::abs(default_output_data[i]) * RelativeTolerance(),
                       1.0f));
    }
  } else {
    for (size_t i = 0; i < output_size; i++) {
      ASSERT_LE(std::abs(default_output_data[i] - delegate_output_data[i]), 1)
          << "default " << +default_output_data[i] << ", delegate "
          << +delegate_output_data[i] << " at index " << i << " / "
          << output_size;
    }
  }
}

void ReduceTester::Test(tflite::BuiltinOperator reduce_op,
                        TfLiteDelegate* delegate) const {
  std::vector<char> buffer = CreateTfLiteModel(reduce_op);
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

  switch (Quantization()) {
    case Quantization::None:
      Test<float>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case Quantization::Signed:
      Test<int8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
    case Quantization::Unsigned:
      Test<uint8_t>(delegate_interpreter.get(), default_interpreter.get());
      break;
  }
}

namespace {

TensorType GetTensorType(enum ReduceTester::Quantization q) {
  switch (q) {
    case ReduceTester::Quantization::None:
      return TensorType_FLOAT32;
    case ReduceTester::Quantization::Signed:
      return TensorType_INT8;
    case ReduceTester::Quantization::Unsigned:
      return TensorType_UINT8;
  }
}

}  // namespace

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

  const flatbuffers::Offset<QuantizationParameters> input_quantization =
      Quantization() == Quantization::None
          ? 0
          : CreateQuantizationParameters(
                builder, /*min=*/0, /*max=*/0,
                builder.CreateVector<float>({InputScale()}),
                builder.CreateVector<int64_t>({InputZeroPoint()}));
  const flatbuffers::Offset<QuantizationParameters> output_quantization =
      Quantization() == Quantization::None
          ? 0
          : CreateQuantizationParameters(
                builder, /*min=*/0, /*max=*/0,
                builder.CreateVector<float>({OutputScale()}),
                builder.CreateVector<int64_t>({OutputZeroPoint()}));
  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(InputShape().data(),
                                                 InputShape().size()),
                   GetTensorType(Quantization()), /*buffer=*/0, /*name=*/0,
                   input_quantization),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(axes_shape.data(), axes_shape.size()),
          TensorType_INT32, /*buffer=*/1),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape.data(),
                                                 output_shape.size()),
                   GetTensorType(Quantization()), /*buffer=*/0, /*name=*/0,
                   output_quantization),
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

  std::string model_description = "Reduce model";
  if (Quantization() != Quantization::None) {
    model_description = "Quantized reduce model";
  }
  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString(model_description);

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
