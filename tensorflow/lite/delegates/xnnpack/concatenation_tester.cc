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

#include "tensorflow/lite/delegates/xnnpack/concatenation_tester.h"

#include <algorithm>
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
#include "tensorflow/compiler/mlir/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

std::vector<int32_t> SameShapeDifferentAxis(std::vector<int32_t> shape,
                                            int axis, int32_t size) {
  std::vector<int32_t> new_shape{shape};
  new_shape[axis < 0 ? axis + shape.size() : axis] = size;
  return new_shape;
}

template <class T>
void ConcatenationTester::Test(Interpreter *delegate_interpreter,
                               Interpreter *default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_int_distribution<int32_t> input_distribution(
      std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  for (size_t i = 0; i < NumInputs(); i++) {
    T *default_input_data = default_interpreter->typed_input_tensor<T>(i);
    std::generate_n(default_input_data, ComputeSize(InputShape(i)),
                    std::ref(input_rng));

    T *xnnpack_input_data = delegate_interpreter->typed_input_tensor<T>(i);
    std::copy_n(default_input_data, ComputeSize(InputShape(i)),
                xnnpack_input_data);
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  T *default_output_data = default_interpreter->typed_output_tensor<T>(0);
  T *xnnpack_output_data = delegate_interpreter->typed_output_tensor<T>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(static_cast<int32_t>(default_output_data[i]),
              static_cast<int32_t>(xnnpack_output_data[i]));
  }
}

template <>
void ConcatenationTester::Test<float>(Interpreter *delegate_interpreter,
                                      Interpreter *default_interpreter) const {
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  std::uniform_real_distribution<float> input_distribution(-25.0f, 25.0f);
  auto input_rng = std::bind(input_distribution, std::ref(rng));

  for (size_t i = 0; i < NumInputs(); i++) {
    float *default_input_data =
        default_interpreter->typed_input_tensor<float>(i);
    std::generate_n(default_input_data, ComputeSize(InputShape(i)),
                    std::ref(input_rng));

    float *xnnpack_input_data =
        delegate_interpreter->typed_input_tensor<float>(i);
    std::copy_n(default_input_data, ComputeSize(InputShape(i)),
                xnnpack_input_data);
  }

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float *default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  float *xnnpack_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  for (size_t i = 0; i < ComputeSize(OutputShape()); i++) {
    ASSERT_EQ(default_output_data[i], xnnpack_output_data[i]);
  }
}

void ConcatenationTester::Test(TensorType tensor_type,
                               TfLiteDelegate *delegate) const {
  std::vector<char> buffer = CreateTfLiteModel(tensor_type);
  const Model *model = GetModel(buffer.data());

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
  ASSERT_EQ(delegate_interpreter->inputs().size(), NumInputs());
  ASSERT_EQ(default_interpreter->inputs().size(), NumInputs());
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

std::vector<char> ConcatenationTester::CreateTfLiteModel(
    TensorType tensor_type) const {
  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<OperatorCode> operator_code =
      CreateOperatorCode(builder, BuiltinOperator_CONCATENATION, 0);

  std::vector<flatbuffers::Offset<Buffer>> buffers{{
      CreateBuffer(builder, builder.CreateVector({})),
  }};

  std::vector<flatbuffers::Offset<Tensor>> tensors;
  tensors.reserve(NumInputs());
  for (size_t i = 0; i < NumInputs(); i++) {
    tensors.push_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(InputShape(i).data(),
                                      InputShape(i).size()),
        tensor_type,
        /*buffer=*/0, /*name=*/0,
        CreateQuantizationParameters(
            builder, /*min=*/0, /*max=*/0,
            builder.CreateVector<float>({input_scales_[i]}),
            builder.CreateVector<int64_t>({input_zero_points_[i]}))));
  }

  tensors.push_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(OutputShape().data(), OutputShape().size()),
      tensor_type,
      /*buffer=*/0, /*name=*/0,
      CreateQuantizationParameters(
          builder, /*min=*/0, /*max=*/0,
          builder.CreateVector<float>({output_scale_}),
          builder.CreateVector<int64_t>({output_zero_point_}))));

  std::vector<int32_t> op_inputs;
  op_inputs.reserve(NumInputs());
  for (size_t i = 0; i < NumInputs(); i++) {
    op_inputs.push_back(static_cast<int32_t>(i));
  }

  const std::array<int32_t, 1> op_outputs{static_cast<int32_t>(NumInputs())};
  BuiltinOptions builtin_options_type = tflite::BuiltinOptions_NONE;
  flatbuffers::Offset<void> builtin_options = 0;
  builtin_options_type = tflite::BuiltinOptions_ConcatenationOptions;
  builtin_options = CreateConcatenationOptions(builder, Axis()).Union();
  const flatbuffers::Offset<Operator> op = CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      builtin_options_type, builtin_options);

  const std::vector<int32_t> subgraph_inputs = op_inputs;
  const std::array<int32_t, 1> subgraph_outputs = op_outputs;
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
      builder.CreateString("Concatenation model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

int32_t ConcatenationTester::ComputeSize(const std::vector<int32_t> &shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), 1,
                         std::multiplies<int32_t>());
}

}  // namespace xnnpack
}  // namespace tflite
