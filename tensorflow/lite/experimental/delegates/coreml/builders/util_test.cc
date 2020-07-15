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
#include "tensorflow/lite/experimental/delegates/coreml/builders/util.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"

using tflite::delegates::coreml::IsBinaryOpSupported;

namespace {

class IsBinaryOpSupportedTest : public testing::Test {
 protected:
  void SetUp() override {
    const int input_size = 2;
    tensors_.resize(input_size);
    context_.tensors = tensors_.data();
    node_.inputs = TfLiteIntArrayCreate(input_size);
    for (int i = 0; i < input_size; ++i) {
      node_.inputs->data[i] = i;
    }

    for (auto& tensor : tensors_) {
      tensor.allocation_type = kTfLiteArenaRw;
      tensor.dims = nullptr;
    }
  }

  void TearDown() override {
    FreeInputShapes();
    TfLiteIntArrayFree(node_.inputs);
  }

  void SetInputShapes(const std::vector<std::vector<int>>& shapes) {
    for (int i = 0; i < tensors_.size(); ++i) {
      tensors_[i].dims = TfLiteIntArrayCreate(shapes[i].size());
      std::copy(shapes[i].begin(), shapes[i].end(), tensors_[i].dims->data);
    }
  }

  void FreeInputShapes() {
    for (auto& tensor : tensors_) {
      if (tensor.dims != nullptr) {
        TfLiteIntArrayFree(tensor.dims);
        tensor.dims = nullptr;
      }
    }
  }

  TfLiteContext context_;
  TfLiteNode node_;
  std::vector<TfLiteTensor> tensors_;
};

TEST_F(IsBinaryOpSupportedTest, BroadcastTest) {
  std::vector<int> base_shape = {2, 2, 3, 4};
  std::vector<std::vector<int>> shapes = {
      {2, 2, 3, 4}, {2, 1, 1, 4}, {2, 2, 3, 1}, {2, 1, 1, 1}};
  std::vector<TfLiteTensor> inputs(2);
  for (const auto& shape : shapes) {
    SetInputShapes({base_shape, shape});
    ASSERT_TRUE(IsBinaryOpSupported(nullptr, &node_, &context_));
    FreeInputShapes();
  }
}

TEST_F(IsBinaryOpSupportedTest, LessThan4DTest) {
  std::vector<int> base_shape = {1, 2, 3, 4};
  std::vector<std::vector<int>> shapes = {{4}, {2, 3, 1}, {1, 1, 1, 1}};
  for (const auto& shape : shapes) {
    SetInputShapes({base_shape, shape});
    ASSERT_TRUE(IsBinaryOpSupported(nullptr, &node_, &context_));
    FreeInputShapes();
  }
}

TEST_F(IsBinaryOpSupportedTest, ConstScalarTest) {
  std::vector<int> base_shape = {2, 2, 3, 4};
  tensors_[1].allocation_type = kTfLiteMmapRo;
  SetInputShapes({base_shape, {1}});
  ASSERT_TRUE(IsBinaryOpSupported(nullptr, &node_, &context_));
  FreeInputShapes();
}

TEST_F(IsBinaryOpSupportedTest, NotSupportedBroadcastTest) {
  std::vector<int> base_shape = {2, 2, 3, 4};
  std::vector<std::vector<int>> shapes = {
      {2, 2, 1, 4}, {2, 1, 2, 4}, {1, 2, 3, 1}, {1, 1, 1, 1}};
  for (const auto& shape : shapes) {
    SetInputShapes({base_shape, shape});
    ASSERT_FALSE(IsBinaryOpSupported(nullptr, &node_, &context_));
    FreeInputShapes();
  }
}
}  // namespace
