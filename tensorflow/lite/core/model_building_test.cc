/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/model_building.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/interpreter.h"

namespace tflite {
namespace model_builder {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::Pointwise;

TEST(ModelBuilderTest,
     SingleGraphDefinitionWithoutFinalBuildDoesNotLeakMemory) {
  ModelBuilder builder;
  Graph graph = NewGraph(builder);
  Tensor in1 = NewInput(graph, kTfLiteInt32);
  Tensor in2 = NewInput(graph, kTfLiteInt32);
  Tensor out = Add(in1, in2);
  MarkOutput(out);
}

struct BinaryOpTestParam {
  std::string name;
  std::function<Tensor(Tensor, Tensor)> op;
  std::vector<int> expected;
};

class ModelBuilderBinaryOpTest
    : public testing::TestWithParam<BinaryOpTestParam> {};

TEST_P(ModelBuilderBinaryOpTest, Works) {
  ModelBuilder builder;
  Graph graph = NewGraph(builder);
  auto [in0, in1] = NewInputs<2>(graph, kTfLiteInt32);
  Tensor out = GetParam().op(in0, in1);
  MarkOutput(out);

  tflite::Interpreter interpreter;
  builder.Build(interpreter);

  ASSERT_THAT(interpreter.inputs().size(), Eq(2));

  interpreter.ResizeInputTensor(0, {2, 3});
  interpreter.ResizeInputTensor(1, {2, 3});
  interpreter.AllocateTensors();

  absl::Span<int32_t> input0(
      reinterpret_cast<int32_t*>(interpreter.input_tensor(0)->data.data), 6);
  absl::Span<int32_t> input1(
      reinterpret_cast<int32_t*>(interpreter.input_tensor(1)->data.data), 6);

  absl::c_iota(input0, -3);
  absl::c_iota(input1, -3);

  interpreter.Invoke();

  absl::Span<int32_t> output(
      reinterpret_cast<int32_t*>(interpreter.output_tensor(0)->data.data), 6);

  EXPECT_THAT(output, ElementsAreArray(GetParam().expected));
}

#define BINARY_PARAM(FUNC, ...)                                          \
  BinaryOpTestParam {                                                    \
    #FUNC, [](auto&&... args) { return FUNC(args...); }, { __VA_ARGS__ } \
  }

INSTANTIATE_TEST_SUITE_P(Blah, ModelBuilderBinaryOpTest,
                         testing::Values(BINARY_PARAM(Add, -6, -4, -2, 0, 2, 4),
                                         BINARY_PARAM(Mul, 9, 4, 1, 0, 1, 4)));

TEST(ModelBuilderTest, AbsGraphWorks) {
  ModelBuilder builder;
  Graph graph = NewGraph(builder);
  const Tensor in0 = NewInput(graph, kTfLiteInt32);
  const Tensor out = Abs(in0);
  MarkOutput(out);

  tflite::Interpreter interpreter;
  builder.Build(interpreter);

  ASSERT_THAT(interpreter.inputs().size(), Eq(1));

  interpreter.ResizeInputTensor(0, {2, 3});
  interpreter.AllocateTensors();

  absl::Span<int32_t> input(
      reinterpret_cast<int32_t*>(interpreter.input_tensor(0)->data.data), 6);

  absl::c_iota(input, -3);

  interpreter.Invoke();

  absl::Span<int32_t> output(
      reinterpret_cast<int32_t*>(interpreter.output_tensor(0)->data.data), 6);

  EXPECT_THAT(output, ElementsAre(3, 2, 1, 0, 1, 2));
}

TEST(ModelBuilderTest, ConstantTensorsCanBeAdded) {
  ModelBuilder builder;
  Buffer buffer0 = NewConstantBuffer<kTfLiteInt32>(
      builder, /*shape=*/{2, 1},
      /*data=*/std::vector<int>{3, 4}, NoQuantization());
  Buffer buffer1 = NewConstantBuffer<kTfLiteInt32>(
      builder, /*shape=*/{2, 1},
      /*data=*/std::vector<int>{7, 3}, NoQuantization());

  Graph graph = NewGraph(builder);
  const Tensor in0 = NewConstantTensor(graph, buffer0);
  const Tensor in1 = NewConstantTensor(graph, buffer1);
  const Tensor out = Add(in0, in1);
  MarkOutput(out);

  tflite::Interpreter interpreter;
  builder.Build(interpreter);

  interpreter.AllocateTensors();
  interpreter.Invoke();

  absl::Span<int32_t> output(
      reinterpret_cast<int32_t*>(interpreter.output_tensor(0)->data.data), 2);

  EXPECT_THAT(output, ElementsAre(10, 7));
}

TEST(ModelBuilderTest, FullyConnectedWorks) {
  ModelBuilder builder;
  Buffer input = NewConstantBuffer<kTfLiteFloat32>(
      builder, /*shape=*/{3, 2},
      /*data=*/std::vector<int>{1, 2, 3, 4, 5, 6}, NoQuantization());
  Buffer weights = NewConstantBuffer<kTfLiteFloat32>(
      builder, /*shape=*/{1, 2},
      /*data=*/std::vector<int>{7, 3}, NoQuantization());

  Graph graph = NewGraph(builder);
  const Tensor in0 = NewConstantTensor(graph, input);
  const Tensor out = FullyConnected(in0, weights);
  MarkOutput(out);

  tflite::Interpreter interpreter;
  builder.Build(interpreter);

  interpreter.AllocateTensors();
  interpreter.Invoke();

  absl::Span<float> output(
      reinterpret_cast<float*>(interpreter.output_tensor(0)->data.data), 3);

  EXPECT_THAT(output, Pointwise(FloatEq(), {13, 33, 53}));
}

}  // namespace
}  // namespace model_builder
}  // namespace tflite
