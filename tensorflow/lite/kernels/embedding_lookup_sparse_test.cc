/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
// Unit test for TFLite sparse lookup op.

#include <cmath>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class EmbeddingLookupSparseOpModel : public SingleOpModel {
 public:
  EmbeddingLookupSparseOpModel(CombinerType type,
                               std::initializer_list<int> lookup_shape,
                               std::initializer_list<int> indices_shape,
                               std::initializer_list<int> dense_shape_shape,
                               std::initializer_list<int> value_shape) {
    lookup_ = AddInput(TensorType_INT32);
    indices_ = AddInput(TensorType_INT32);
    dense_shape_ = AddInput(TensorType_INT32);
    weights_ = AddInput(TensorType_FLOAT32);
    value_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_EMBEDDING_LOOKUP_SPARSE,
                 BuiltinOptions_EmbeddingLookupSparseOptions,
                 CreateEmbeddingLookupSparseOptions(builder_, type).Union());
    BuildInterpreter({lookup_shape, indices_shape, dense_shape_shape,
                      lookup_shape, value_shape});
  }

  void SetInput(std::initializer_list<int> lookup_data,
                std::initializer_list<int> indices_data,
                std::initializer_list<int> dense_shape_data,
                std::initializer_list<float> weights_data) {
    PopulateTensor(lookup_, lookup_data);
    PopulateTensor(indices_, indices_data);
    PopulateTensor(dense_shape_, dense_shape_data);
    PopulateTensor(weights_, weights_data);
  }

  void Set3DWeightMatrix(const std::function<float(int, int, int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(value_);
    int rows = tensor->dims->data[0];
    int columns = tensor->dims->data[1];
    int features = tensor->dims->data[2];
    float* tensor_ptr = GetTensorData<float>(tensor);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        for (int k = 0; k < features; k++) {
          tensor_ptr[(i * columns + j) * features + k] = function(i, j, k);
        }
      }
    }
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int lookup_;
  int weights_;
  int indices_;
  int dense_shape_;
  int value_;
  int output_;
};

TEST(EmbeddingLookupSparseOpTest, SimpleTest) {
  EmbeddingLookupSparseOpModel m(CombinerType_SUM, {3}, {3, 2}, {2}, {4, 3, 2});
  m.SetInput({1, 3, 0}, {0, 0, 2, 0, 2, 1}, {3, 2}, {1.0, 2.0, 4.0});
  m.Set3DWeightMatrix(
      [](int i, int j, int k) { return i + j / 10.0f + k / 100.0f; });
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.10, 1.11, 1.20, 1.21,  // Row 1
                  0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  // -
                  6.00, 6.06, 6.60, 6.66, 7.20, 7.26,  // 2 * Row 3 + 4 * Row 0
              })));
}

TEST(EmbeddingLookupSparseOpTest, SimpleTestMean) {
  EmbeddingLookupSparseOpModel m(CombinerType_MEAN, {3}, {3, 2}, {2},
                                 {4, 3, 2});
  m.SetInput({1, 3, 0}, {0, 0, 2, 0, 2, 1}, {3, 2}, {1.0, 2.0, 4.0});
  m.Set3DWeightMatrix(
      [](int i, int j, int k) { return i + j / 10.0f + k / 100.0f; });
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.10, 1.11, 1.20, 1.21,  // Row 1
                  0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  // -
                  1.00, 1.01, 1.10, 1.11, 1.20, 1.21,  // 2 * Row 3 + 4 * Row 0
              })));
}

TEST(EmbeddingLookupSparseOpTest, SimpleTestSqrtn) {
  EmbeddingLookupSparseOpModel m(CombinerType_SQRTN, {3}, {3, 2}, {2},
                                 {4, 3, 2});
  m.SetInput({1, 3, 0}, {0, 0, 2, 0, 2, 1}, {3, 2}, {1.0, 2.0, 4.0});
  m.Set3DWeightMatrix(
      [](int i, int j, int k) { return i + j / 10.0f + k / 100.0f; });
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.10, 1.11, 1.20, 1.21,  // Row 1
                  0.00, 0.00, 0.00, 0.00, 0.00, 0.00,  // -
                  6.00f / std::sqrt(20.0f), 6.06f / std::sqrt(20.0f),
                  6.60f / std::sqrt(20.0f), 6.66f / std::sqrt(20.0f),
                  7.20f / std::sqrt(20.0f),
                  7.26f / std::sqrt(20.0f),  // 2 * Row 3 + 4 * Row 0,  // 2 *
                                             // Row 3 + 4 * Row 0
              })));
}

TEST(EmbeddingLookupSparseOpTest, Indices3DTest) {
  EmbeddingLookupSparseOpModel m(CombinerType_SUM, {3}, {3, 3}, {3}, {4, 3, 2});
  m.SetInput({1, 3, 0}, {0, 0, 0, 2, 0, 0, 2, 0, 1}, {3, 2, 2},
             {1.0, 2.0, 4.0});
  m.Set3DWeightMatrix(
      [](int i, int j, int k) { return i + j / 10.0f + k / 100.0f; });
  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.10, 1.11, 1.20, 1.21, 0.00, 0.00, 0.00,
                  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
                  0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 6.00, 6.06, 6.60,
                  6.66, 7.20, 7.26, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
              })));
}

}  // namespace
}  // namespace tflite
