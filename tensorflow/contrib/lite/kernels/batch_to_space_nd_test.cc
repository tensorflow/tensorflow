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

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BatchToSpaceNDOpModel : public SingleOpModel {
 public:
  BatchToSpaceNDOpModel(std::initializer_list<int> input_shape,
                        std::initializer_list<int> block_shape,
                        std::initializer_list<int> before_crops,
                        std::initializer_list<int> after_crops) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(
                     builder_, builder_.CreateVector<int>(block_shape),
                     builder_.CreateVector<int>(before_crops),
                     builder_.CreateVector<int>(after_crops))
                     .Union());
    BuildInterpreter({input_shape});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(BatchToSpaceNDOpTest, SimpleTest) {
  BatchToSpaceNDOpModel m({4, 2, 2, 1}, {2, 2}, {0, 0}, {0, 0});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 5, 2, 6, 9, 13, 10, 14, 3, 7,
                                               4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, InvalidShapeTest) {
  EXPECT_DEATH(BatchToSpaceNDOpModel({3, 2, 2, 1}, {2, 2}, {0, 0}, {0, 0}),
               "Cannot allocate tensors");
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
