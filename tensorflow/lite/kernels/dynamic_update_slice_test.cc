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
#include <stdint.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/subgraph_test_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class DynamicUpdateSliceOpModel : public SingleOpModel {
 public:
  DynamicUpdateSliceOpModel(const TensorData& operand, const TensorData& update,
                            const TensorData& start_indices) {
    input_ = AddInput(operand);
    update_ = AddInput(update);
    start_indices_ = AddInput(start_indices);
    output_ = AddOutput(operand.type);
    SetBuiltinOp(BuiltinOperator_DYNAMIC_UPDATE_SLICE,
                 BuiltinOptions_DynamicUpdateSliceOptions,
                 CreateDynamicUpdateSliceOptions(builder_).Union());
    BuildInterpreter(
        {GetShape(input_), GetShape(update_), GetShape(start_indices_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  template <typename T>
  void SetUpdate(std::initializer_list<T> data) {
    PopulateTensor<T>(update_, data);
  }

  void SetStringInput(std::initializer_list<string> data) {
    PopulateStringTensor(input_, data);
  }

  template <typename T>
  void SetStartIndices(std::initializer_list<T> data) {
    PopulateTensor<T>(start_indices_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<string> GetStringOutput() {
    return ExtractVector<string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int update_;
  int start_indices_;
  int output_;
};

TEST(DynamicUpdateSliceOpTest, SimpleTestF32InPlaceInput) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  const int kInplaceInputTensorIdx = 0;
  const int kInplaceOutputTensorIdx = 0;
  const TfLiteTensor* input_tensor = m.GetInputTensor(kInplaceInputTensorIdx);
  TfLiteTensor* output_tensor = m.GetOutputTensor(kInplaceOutputTensorIdx);
  output_tensor->data.data = input_tensor->data.data;
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,   //
                                               4, -1, 6,  //
                                               7, -2, 9})));
  EXPECT_EQ(output_tensor->data.data, input_tensor->data.data);
}

TEST(DynamicUpdateSliceOpTest, SimpleTestF32) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,   //
                                               4, -1, 6,  //
                                               7, -2, 9})));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI1) {
  DynamicUpdateSliceOpModel m({TensorType_BOOL, {3, 3}},
                              {TensorType_BOOL, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<bool>({true, true, true,  //
                    true, true, true,  //
                    true, true, true});
  m.SetUpdate<bool>({false, false});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<bool>(), ElementsAreArray({true, true, true,   //
                                                     true, false, true,  //
                                                     true, false, true}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI8) {
  DynamicUpdateSliceOpModel m({TensorType_INT8, {3, 3}},
                              {TensorType_INT8, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int8_t>({1, 2, 3,  //
                      4, 5, 6,  //
                      7, 8, 9});
  m.SetUpdate<int8_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({1, 2, 3,   //
                                                       4, -1, 6,  //
                                                       7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI32) {
  DynamicUpdateSliceOpModel m({TensorType_INT32, {3, 3}},
                              {TensorType_INT32, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int32_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetUpdate<int32_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({1, 2, 3,   //
                                                        4, -1, 6,  //
                                                        7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, ZeroSizeTestI32) {
  DynamicUpdateSliceOpModel m({TensorType_INT32, {3, 3}},
                              {TensorType_INT32, {2, 0}},
                              {TensorType_INT32, {2}});
  m.SetInput<int32_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int32_t>(), ElementsAreArray({1, 2, 3,  //
                                                        4, 5, 6,  //
                                                        7, 8, 9}));
}

TEST(DynamicUpdateSliceOpTest, SimpleTestI64) {
  DynamicUpdateSliceOpModel m({TensorType_INT64, {3, 3}},
                              {TensorType_INT64, {2, 1}},
                              {TensorType_INT32, {2}});
  m.SetInput<int64_t>({1, 2, 3,  //
                       4, 5, 6,  //
                       7, 8, 9});
  m.SetUpdate<int64_t>({-1, -2});
  m.SetStartIndices<int32_t>({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<int64_t>(), ElementsAreArray({1, 2, 3,   //
                                                        4, -1, 6,  //
                                                        7, -2, 9}));
}

TEST(DynamicUpdateSliceOpTest, BoundaryTest) {
  DynamicUpdateSliceOpModel m({TensorType_FLOAT32, {3, 3}},
                              {TensorType_FLOAT32, {2, 2}},
                              {TensorType_INT32, {2}});
  m.SetInput<float>({1, 2, 3,  //
                     4, 5, 6,  //
                     7, 8, 9});
  m.SetUpdate<float>({-1, -2,  //
                      -3, -4});
  m.SetStartIndices<int32_t>({2, 2});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({1, 2, 3,    //
                                               4, -1, -2,  //
                                               7, -3, -4})));
}

TEST(DynamicUpdateSliceOpTest, UpdateShapeTooLargeTest) {
  EXPECT_DEATH_IF_SUPPORTED(
      DynamicUpdateSliceOpModel({TensorType_FLOAT32, {3, 3}},
                                {TensorType_FLOAT32, {4, 2}},
                                {TensorType_INT32, {2}}),
      "SizeOfDimension\\(update, i\\) <= SizeOfDimension\\(operand, "
      "i\\) was not true.");
}

// Sets up an interpreter and a graph to test inplace use of input tensors for
// the operation's output.
class DynamicUpdateSliceGraphModel {
 public:
  static constexpr struct InPlaceGraph {
  } kInPlaceGraph{};
  static constexpr struct NotInPlaceGraph {
  } kNotInPlaceGraph{};

  DynamicUpdateSliceGraphModel(InPlaceGraph, bool multiple_consumers) {
    builder_.BuildInplaceDynamicUpdateSliceSubgraph(
        interpreter_.primary_subgraph(), multiple_consumers);
    SetUpInterpreter();
  }

  explicit DynamicUpdateSliceGraphModel(NotInPlaceGraph) {
    builder_.BuildInputDynamicUpdateSliceSubgraph(
        interpreter_.primary_subgraph());
    SetUpInterpreter();
  }

  void SetUpInterpreter() {
    interpreter_.ResizeInputTensor(interpreter_.inputs()[0], {2, 3});
    interpreter_.ResizeInputTensor(interpreter_.inputs()[1], {1, 3});
    interpreter_.ResizeInputTensor(interpreter_.inputs()[2], {2});
    CHECK_EQ(interpreter_.AllocateTensors(), kTfLiteOk);
    subgraph_test_util::FillIntTensor(&GetInputTensor(0), {0, 0, 0, 0, 0, 0});
    subgraph_test_util::FillIntTensor(&GetInputTensor(1), {3, 3, 3});
    subgraph_test_util::FillIntTensor(&GetInputTensor(2), {1, 0});
  }

  Interpreter& GetInterpreter() { return interpreter_; }

  // Get a tensor given its internal index.
  TfLiteTensor& GetTensor(int index) { return *interpreter_.tensor(index); }

  // Get an input tensor given the declaration order.
  TfLiteTensor& GetInputTensor(int index) {
    return GetTensor(interpreter_.inputs()[index]);
  }

  // Get an output tensor given the declaration order.
  TfLiteTensor& GetOutputTensor(int index) {
    return GetTensor(interpreter_.outputs()[index]);
  }

 protected:
  Interpreter interpreter_;
  // The builder serves as an RAII guard for some of the tensor buffers and must
  // live until the end of the test.
  subgraph_test_util::SubgraphBuilder builder_;
};

absl::Span<int> ShapeOf(const TfLiteTensor& tensor) {
  if (!tensor.dims) {
    return {};
  }
  return absl::Span<int>(tensor.dims->data, tensor.dims->size);
}

template <class T>
absl::Span<int32_t> DataOf(const TfLiteTensor& tensor) {
  return absl::Span<int>(tensor.data.i32, tensor.bytes / sizeof(T));
}

TEST(DynamicUpdateSliceOpTest, DoNotReuseGraphInputBuffer) {
  auto model = DynamicUpdateSliceGraphModel(
      DynamicUpdateSliceGraphModel::kNotInPlaceGraph);
  ASSERT_EQ(model.GetInterpreter().Invoke(), kTfLiteOk);

  const TfLiteTensor& output = model.GetOutputTensor(0);
  EXPECT_THAT(ShapeOf(output), ElementsAre(2, 3));
  EXPECT_THAT(DataOf<int32_t>(output), ElementsAre(1, 1, 1, 4, 4, 4));

  const TfLiteTensor& input0 = model.GetInputTensor(0);
  const TfLiteTensor& intermediate = model.GetTensor(5);
  EXPECT_NE(input0.data.data, intermediate.data.data);
}

TEST(DynamicUpdateSliceOpTest, OnlyShareBufferForASingleConsumer) {
  for (bool multiple_consumers : {true, false}) {
    auto model = DynamicUpdateSliceGraphModel(
        DynamicUpdateSliceGraphModel::kInPlaceGraph, multiple_consumers);
    ASSERT_EQ(model.GetInterpreter().Invoke(), kTfLiteOk);

    const TfLiteTensor& output = model.GetOutputTensor(0);
    EXPECT_THAT(ShapeOf(output), ElementsAre(2, 3));
    EXPECT_THAT(DataOf<int32_t>(output), ElementsAre(2, 2, 2, 4, 4, 4));

    const TfLiteTensor& intermediate0 = model.GetTensor(5);
    const TfLiteTensor& intermediate1 = model.GetTensor(6);
    if (multiple_consumers) {
      EXPECT_NE(intermediate0.data.data, intermediate1.data.data);
    } else {
      EXPECT_EQ(intermediate0.data.data, intermediate1.data.data);
    }
  }
}

}  // namespace
}  // namespace tflite
