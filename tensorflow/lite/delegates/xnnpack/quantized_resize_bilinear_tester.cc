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

#include "tensorflow/lite/delegates/xnnpack/quantized_resize_bilinear_tester.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
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
void QuantizedResizeBilinearTester::Test(
    Interpreter* delegate_interpreter, Interpreter* default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto input_rng = std::bind(
      std::uniform_int_distribution<int32_t>(std::numeric_limits<T>::min(),
                                             std::numeric_limits<T>::max()),
      std::ref(rng));

  T* default_input_data = default_interpreter->typed_input_tensor<T>(0);
  std::generate_n(default_input_data,
                  BatchSize() * InputHeight() * InputWidth() * Channels(),
                  std::ref(input_rng));

  T* delegate_input_data = delegate_interpreter->typed_input_tensor<T>(0);
  std::copy_n(default_input_data,
              BatchSize() * InputHeight() * InputWidth() * Channels(),
              delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T* default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T* delegate_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (int i = 0; i < BatchSize(); i++) {
    for (int y = 0; y < OutputHeight(); y++) {
      for (int x = 0; x < OutputWidth(); x++) {
        for (int c = 0; c < Channels(); c++) {
          const int index =
              ((i * OutputHeight() + y) * OutputWidth() + x) * Channels() + c;
          ASSERT_LE(std::abs(static_cast<int32_t>(default_output_data[index]) -
                             static_cast<int32_t>(delegate_output_data[index])),
                    1)
              << "batch " << i << " / " << BatchSize() << ", y position " << y
              << " / " << OutputHeight() << ", x position " << x << " / "
              << OutputWidth() << ", channel " << c << " / " << Channels();
        }
      }
    }
  }
}

void QuantizedResizeBilinearTester::Test(TfLiteDelegate* delegate) const {
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

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

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

std::vector<char> QuantizedResizeBilinearTester::CreateTfLiteModel() const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_RESIZE_BILINEAR);

  flatbuffers::Offset<tflite::ResizeBilinearOptions> resize_bilinear_options =
      CreateResizeBilinearOptions(builder, AlignCorners(), HalfPixelCenters());

  const std::array<int32_t, 2> size_data{{OutputHeight(), OutputWidth()}};

  const std::array<flatbuffers::Offset<Buffer>, 2> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
      CreateBuffer(builder,
                   builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(size_data.data()),
                       size_data.size() * sizeof(int32_t))),
  }};

  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), Channels()}};
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), Channels()}};
  const std::array<int32_t, 1> size_shape{
      {static_cast<int32_t>(size_data.size())}};

  const std::array<flatbuffers::Offset<Tensor>, 3> tensors{{
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
          Unsigned() ? TensorType_UINT8 : TensorType_INT8,
          /*buffer=*/0, /*name=*/0,
          CreateQuantizationParameters(
              builder, /*min=*/0, /*max=*/0,
              builder.CreateVector<float>({Scale()}),
              builder.CreateVector<int64_t>({ZeroPoint()}))),
      CreateTensor(
          builder,
          builder.CreateVector<int32_t>(size_shape.data(), size_shape.size()),
          TensorType_INT32, /*buffer=*/1),
      CreateTensor(builder,
                   builder.CreateVector<int32_t>(output_shape.data(),
                                                 output_shape.size()),
                   Unsigned() ? TensorType_UINT8 : TensorType_INT8,
                   /*buffer=*/0, /*name=*/0,
                   CreateQuantizationParameters(
                       builder, /*min=*/0, /*max=*/0,
                       builder.CreateVector<float>({Scale()}),
                       builder.CreateVector<int64_t>({ZeroPoint()}))),
  }};

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_ResizeBilinearOptions, resize_bilinear_options.Union());

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
      builder.CreateString("Quantized Resize Bilinear model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
