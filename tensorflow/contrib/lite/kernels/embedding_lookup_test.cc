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
// Unit test for TFLite Lookup op.

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

class EmbeddingLookupOpModel : public SingleOpModel {
 public:
  EmbeddingLookupOpModel(std::initializer_list<int> index_shape,
                         std::initializer_list<int> weight_shape) {
    input_ = AddInput(TensorType_INT32);
    weight_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_EMBEDDING_LOOKUP, BuiltinOptions_NONE, 0);
    BuildInterpreter({index_shape, weight_shape});
  }

  void SetInput(std::initializer_list<int> data) {
    PopulateTensor(input_, data);
  }

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

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int weight_;
  int output_;
};

// TODO(ahentz): write more tests that exercise the details of the op, such as
// lookup errors and variable input shapes.
TEST(EmbeddingLookupOpTest, SimpleTest) {
  EmbeddingLookupOpModel m({3}, {3, 2, 4});
  m.PopulateTensor<int>(0, {1, 0, 2});
  m.Set3DWeightMatrix(
      [](int i, int j, int k) { return i + j / 10.0f + k / 100.0f; });

  m.Invoke();

  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray(ArrayFloatNear({
                  1.00f, 1.01f, 1.02f, 1.03f, 1.10f, 1.11f, 1.12f, 1.13f,  // Row 1
                  0.00f, 0.01f, 0.02f, 0.03f, 0.10f, 0.11f, 0.12f, 0.13f,  // Row 0
                  2.00f, 2.01f, 2.02f, 2.03f, 2.10f, 2.11f, 2.12f, 2.13f,  // Row 2
              })));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
