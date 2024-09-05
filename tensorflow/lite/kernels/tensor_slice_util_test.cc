/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/tensor_slice_util.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace {

using ::testing::ElementsAreArray;

TEST(TensorSliceUtil, ArrayContains) {
  std::vector<int64_t> array = {1, 2, 3};

  EXPECT_TRUE(ArrayContains(array.data(), array.size(), 2));
  EXPECT_FALSE(ArrayContains(array.data(), array.size(), 0));
}

TEST(TensorSliceUtil, ArrayContainsWorkOnEmptyArray) {
  std::vector<int64_t> array = {};

  EXPECT_FALSE(ArrayContains(array.data(), 0, 2));
}

TEST(TensorSliceUtil, ScatterIndexHandlesNullPtr) {
  Index<int64_t> index = {3, 5};
  std::vector<int64_t> scatter_dims = {1, 0};
  Index<int64_t>* result = nullptr;
  TfLiteStatus status =
      ScatterIndex(index, scatter_dims.data(), scatter_dims.size(), 3, result);
  EXPECT_THAT(status, kTfLiteError);
}

TEST(TensorSliceUtil, ScatterIndexHandlesOutOfBoundIndices) {
  Index<int64_t> index = {3, 5};
  std::vector<int64_t> scatter_dims = {4, 0};
  Index<int64_t> result;
  TfLiteStatus status =
      ScatterIndex(index, scatter_dims.data(), scatter_dims.size(), 3, &result);

  EXPECT_THAT(status, kTfLiteError);
}

TEST(TensorSliceUtil, ScatterIndex) {
  Index<int64_t> index = {3, 5};
  std::vector<int64_t> scatter_dims = {1, 0};
  Index<int64_t> result;
  ScatterIndex(index, scatter_dims.data(), scatter_dims.size(), 3, &result);

  EXPECT_THAT(result, ElementsAreArray({5, 3, 0}));
}

TEST(TensorSliceUtil, TensorIndexToFlatWorksForScalars) {
  Index<int64_t> index = {0};
  RuntimeShape shape(0);

  EXPECT_EQ(TensorIndexToFlat(index.data(), index.size(), shape), 0);
}

TEST(TensorSliceUtil, TensorIndexToFlat) {
  Index<int64_t> index = {2, 4};
  RuntimeShape shape({3, 5});

  EXPECT_EQ(TensorIndexToFlat(index.data(), index.size(), shape), 14);
}

TEST(TensorSliceUtil, AddIndices) {
  Index<int64_t> index1 = {1, 2, 3};
  Index<int64_t> index2 = {2, 7, 5};
  EXPECT_THAT(AddIndices(index1, index2), ElementsAreArray({3, 9, 8}));
}

TEST(TensorSliceUtil, ExpandDimsHandlesEmptyIndex) {
  Index<int64_t> index = {};
  std::vector<int64_t> avoided_dims = {0, 1};
  Index<int64_t> result;

  ExpandDims(index, avoided_dims.data(), avoided_dims.size(), &result);
  EXPECT_THAT(result, ElementsAreArray({0, 0}));
}

TEST(TensorSliceUtil, ExpandDims) {
  Index<int64_t> index = {2, 4};
  std::vector<int64_t> avoided_dims = {0, 2};
  Index<int64_t> result;

  ExpandDims(index, avoided_dims.data(), avoided_dims.size(), &result);
  EXPECT_THAT(result, ElementsAreArray({0, 2, 0, 4}));
}

TEST(TensorSliceUtil, ReadIndexVector) {
  TfLiteTensor tensor;
  tensor.type = kTfLiteInt64;
  // [
  //   [[0, 2], [1, 0], [2, 1]],
  //   [[0, 1], [1, 0], [0, 9]]
  // ]
  std::vector<int64_t> tensor_data = {0, 2, 1, 0, 2, 1, 0, 1, 1, 0, 0, 9};
  TfLitePtrUnion ptr_union;
  ptr_union.i64 = tensor_data.data();
  tensor.data = ptr_union;

  RuntimeShape shape = {2, 3, 2};
  Index<int64_t> other_indices = {1, 1};
  int64_t dim_to_read = 1;
  EXPECT_THAT(ReadIndexVector(&tensor, shape, other_indices, dim_to_read),
              ElementsAreArray({1, 0, 9}));
}

}  // namespace
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
