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

#include "tensorflow/lite/core/subgraph.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/stderr_reporter.h"
#include "tensorflow/lite/util.h"

namespace tflite {

namespace ops {
namespace builtin {
TfLiteRegistration* Register_PADV2();
TfLiteRegistration* Register_NEG();
}  // namespace builtin
}  // namespace ops

namespace {

using testing::ElementsAreArray;
using testing::Not;

TEST(RemoveUnusedInputs, NothingToRemove) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(4);
  subgraph.SetInputs({0, 1});
  subgraph.SetOutputs({3});
  TfLiteRegistration* pad_op = tflite::ops::builtin::Register_PADV2();
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  subgraph.AddNodeWithParameters({0, 1}, {2}, {}, nullptr, 0, nullptr, pad_op);
  subgraph.AddNodeWithParameters({2}, {3}, {}, nullptr, 0, nullptr, neg_op);

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({0, 1}));
}

TEST(RemoveUnusedInputs, HasUnusedInputs) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(4);
  subgraph.SetInputs({0, 1, 2});
  subgraph.SetOutputs({3});
  TfLiteRegistration* neg_op = tflite::ops::builtin::Register_NEG();
  subgraph.AddNodeWithParameters({2}, {3}, {}, nullptr, 0, nullptr, neg_op);

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({-1, -1, 2}));
}

TEST(RemoveUnusedInputs, BypassInputsWithoutOp) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  subgraph.AddTensors(3);
  subgraph.SetInputs({0, 1, 2});
  subgraph.SetOutputs({0, 2});

  ASSERT_EQ(subgraph.RemoveUnusedInputs(), kTfLiteOk);
  ASSERT_EQ(subgraph.inputs(), std::vector<int>({0, -1, 2}));
}

TEST(GetSubgraphContext, NonConstGetSubgraphContext) {
  Interpreter interpreter;
  auto& subgraph = interpreter.primary_subgraph();
  TfLiteContext* context;

  context = subgraph.GetSubgraphContext(0);
  ASSERT_NE(context, nullptr);

  context = subgraph.GetSubgraphContext(-1);
  ASSERT_EQ(context, nullptr);

  context = subgraph.GetSubgraphContext(1);
  ASSERT_EQ(context, nullptr);

  const auto& const_subgraph = interpreter.primary_subgraph();
  const_subgraph.GetSubgraphContext(1);
}

TEST(GetSubgraphContext, ConstGetSubgraphContext) {
  Interpreter interpreter;
  const auto& subgraph = interpreter.primary_subgraph();
  const TfLiteContext* context;

  context = subgraph.GetSubgraphContext(0);
  ASSERT_NE(context, nullptr);

  context = subgraph.GetSubgraphContext(-1);
  ASSERT_EQ(context, nullptr);

  context = subgraph.GetSubgraphContext(1);
  ASSERT_EQ(context, nullptr);
}

TEST(MarkSubgraphAsDelegationSkippable, MarkSubgraphAsDelegationSkippable) {
  static StderrReporter* error_reporter = new StderrReporter;
  // Construct a mock subgraph vector with two entries.
  std::vector<std::unique_ptr<Subgraph>> subgraphs;
  for (int i = 0; i < 2; ++i) {
    subgraphs.emplace_back(new Subgraph(/*error_reporter=*/error_reporter,
                                        /*external_contexts=*/nullptr,
                                        /*subgraphs=*/&subgraphs,
                                        /*resources=*/nullptr,
                                        /*resource_ids=*/nullptr,
                                        /*initialization_status_map=*/nullptr,
                                        /*subgraph_index=*/i));
  }

  // The primary subgraph shouldn't be delegation-skippable.
  ASSERT_EQ(subgraphs[0]->MarkSubgraphAsDelegationSkippable(0), kTfLiteError);
  ASSERT_FALSE(subgraphs[0]->IsDelegationSkippable());

  // The subgraph_index shouldn't exceed the total number of subgraphs.
  ASSERT_EQ(subgraphs[0]->MarkSubgraphAsDelegationSkippable(2), kTfLiteError);

  ASSERT_EQ(subgraphs[0]->MarkSubgraphAsDelegationSkippable(1), kTfLiteOk);
  ASSERT_TRUE(subgraphs[1]->IsDelegationSkippable());
}

// Helper to get the minimal buffer size to allocate for a buffer of given
// shape.
size_t BytesFor(const TfLiteType type, const int* const data,
                const size_t size) {
  size_t type_size;
  CHECK(GetSizeOfType(nullptr, type, &type_size) == kTfLiteOk)
      << "Type is not supported by GetSizeOfType";
  return std::accumulate(data, data + size, type_size, std::multiplies<int>());
}

size_t BytesFor(const TfLiteType type, const TfLiteIntArray& dims) {
  return BytesFor(type, dims.data, dims.size);
}

size_t BytesFor(const TfLiteType type, const std::vector<int>& dims) {
  return BytesFor(type, dims.data(), dims.size());
}

// Sets up a TFLite context and default values to initialize/resize test
// tensors.
class SubgraphResizeTensorTest : public testing::Test {
 public:
  SubgraphResizeTensorTest() {
    tensor_.type = type_;
    tensor_.allocation_type = kTfLiteDynamic;
  }

  ~SubgraphResizeTensorTest() override { TfLiteTensorFree(&tensor_); }

 protected:
  const TfLiteType type_ = kTfLiteInt32;
  Interpreter interpreter_;
  TfLiteContext& context_ = *interpreter_.primary_subgraph().context();
  const std::vector<int> reference_shape_ = {5, 4, 3};
  const size_t reference_dims_bytes_ = BytesFor(type_, reference_shape_);
  TfLiteTensor tensor_ = {};
  TfLiteIntArray* dims_ = ConvertVectorToTfLiteIntArray(reference_shape_);
};

TEST_F(SubgraphResizeTensorTest, ResizeEmptyDynamicTensorAllocateData) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  EXPECT_EQ(tensor_.dims, dims_);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  // Touch memory to trigger ASAN in case of failure.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeEmptyDynamicTensorWithStoredShapeAllocatesData) {
  tensor_.dims = dims_;
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest, ResizeDynamicTensorWithTheEqualShapeIsANoop) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  const void* const initial_data = tensor_.data.data;

  TfLiteIntArray* dims2 = ConvertVectorToTfLiteIntArray(reference_shape_);
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims2), kTfLiteOk);

  EXPECT_EQ(tensor_.dims, dims2);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_GE(tensor_.data.data, initial_data);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest, ResizeDynamicTensorWithStoredShapeIsANoop) {
  tensor_.dims = dims_;
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);
  const void* const initial_data = tensor_.data.data;
  // Reallocate the tensor with its current shape.
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, tensor_.dims),
            kTfLiteOk);
  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_GE(tensor_.data.data, initial_data);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithEquivalentBufferSizeIsANoop) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  const void* const initial_data = tensor_.data.data;

  const std::vector<int> new_shape = {3, 4, 5};
  ASSERT_THAT(new_shape, Not(ElementsAreArray(reference_shape_)));
  TfLiteIntArray* dims2 = ConvertVectorToTfLiteIntArray(new_shape);
  ASSERT_EQ(BytesFor(type_, *dims2), reference_dims_bytes_);

  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims2), kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, reference_dims_bytes_);
  EXPECT_EQ(tensor_.data.data, initial_data);
  EXPECT_EQ(tensor_.dims, dims2);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, reference_dims_bytes_, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

TEST_F(SubgraphResizeTensorTest,
       ResizeDynamicTensorWithDifferentShapeReallocatesData) {
  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims_), kTfLiteOk);
  const void* const initial_data = tensor_.data.data;

  TfLiteIntArray* dims2 = ConvertVectorToTfLiteIntArray({5, 4, 6});
  const int dims2_bytes = BytesFor(type_, *dims2);
  ASSERT_NE(dims2_bytes, reference_dims_bytes_);

  ASSERT_EQ(context_.ResizeTensor(&context_, &tensor_, dims2), kTfLiteOk);

  // Some alignment requirements may lead to more memory being allocated.
  EXPECT_GE(tensor_.bytes, dims2_bytes);
  EXPECT_NE(tensor_.data.data, initial_data);
  EXPECT_EQ(tensor_.dims, dims2);
  // Touch memory to trigger ASAN in case of incorrect handling.
  std::fill_n(tensor_.data.raw, dims2_bytes, 0);
  std::fill_n(tensor_.dims->data, tensor_.dims->size, 1);
}

}  // namespace
}  // namespace tflite
