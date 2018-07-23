/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the
License.
==============================================================================*/
// Unit test for TFLite Lookup op.

#include <initializer_list>
#include <iomanip>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BaseEmbeddingLookupOpModel : public SingleOpModel {
 public:
  BaseEmbeddingLookupOpModel(std::initializer_list<int> index_shape,
                             std::initializer_list<int> weight_shape,
                             TensorType weight_type = TensorType_FLOAT32) {
    input_ = AddInput(TensorType_INT32);
    weight_ = AddInput(weight_type);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_EMBEDDING_LOOKUP, BuiltinOptions_NONE, 0);
    BuildInterpreter({index_shape, weight_shape});
  }

  void SetInput(std::initializer_list<int> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 protected:
  int input_;
  int weight_;
  int output_;
};

class EmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  using BaseEmbeddingLookupOpModel::BaseEmbeddingLookupOpModel;

  void Set3DWeightMatrix(const std::function<float(int, int, int)>& function) {
    TfLiteTensor* tensor = interpreter_->tensor(weight_);
    int rows = tensor->dims->data[0];
    int columns = tensor->dims->data[1];
    int features = tensor->dims->data[2];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        for (int k = 0; k < features; k++) {
          tensor->data.f[(i * columns + j) * features + k] = function(i, j, k);
        }
      }
    }
  }
};

class HybridEmbeddingLookupOpModel : public BaseEmbeddingLookupOpModel {
 public:
  HybridEmbeddingLookupOpModel(std::initializer_list<int> index_shape,
                               std::initializer_list<int> weight_shape)
      : BaseEmbeddingLookupOpModel(index_shape, weight_shape,
                                   TensorType_UINT8) {}

  void SetWeight(std::initializer_list<float> data) {
    SymmetricQuantizeAndPopulate(weight_, data);
  }
};

// TODO(ahentz): write more tests that exercise the details of the op, such as
// lookup errors and variable input shapes.
TEST(EmbeddingLookupOpTest, SimpleTest) {
  EmbeddingLookupOpModel m({3}, {3, 2, 4});
  m.SetInput({1, 0, 2});
  m.Set3DWeightMatrix(
      [](int i, int j, int k) { return i + j / 10.0f + k / 100.0f; });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.00, 1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                  0.00, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                  2.00, 2.01, 2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
              })));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple2DTest) {
  HybridEmbeddingLookupOpModel m({3}, {3, 8});
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  7.41e-03)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple3DTest) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 4});
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  7.41e-03)));
}

TEST(HybridEmbeddingLookupHybridOpTest, Simple4DTest) {
  HybridEmbeddingLookupOpModel m({3}, {3, 2, 2, 2});
  m.SetInput({1, 0, 2});
  m.SetWeight({
      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
  });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear(
                  {
                      1.00, -1.01, 1.02, 1.03, 1.10, 1.11, 1.12, 1.13,  // Row 1
                      0.00, 0.01,  0.02, 0.03, 0.10, 0.11, 0.12, 0.13,  // Row 0
                      2.00, 2.01,  2.02, 2.03, 2.10, 2.11, 2.12, 2.13,  // Row 2
                  },
                  7.41e-03)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
