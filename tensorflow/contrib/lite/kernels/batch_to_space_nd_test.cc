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
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetBlockShape(std::initializer_list<int> data) {
    PopulateTensor<int>(block_shape_, data);
  }

  void SetCrops(std::initializer_list<int> data) {
    PopulateTensor<int>(crops_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int block_shape_;
  int crops_;
  int output_;
};

// Tests case where block_shape and crops are const tensors.
//
// Example usage is as follows:
//    BatchToSpaceNDOpConstModel m(input_shape, block_shape, crops);
//    m.SetInput(input_data);
//    m.Invoke();
class BatchToSpaceNDOpConstModel : public BatchToSpaceNDOpModel {
 public:
  BatchToSpaceNDOpConstModel(std::initializer_list<int> input_shape,
                             std::initializer_list<int> block_shape,
                             std::initializer_list<int> crops) {
    input_ = AddInput(TensorType_FLOAT32);
    block_shape_ = AddConstInput(TensorType_INT32, block_shape, {2});
    crops_ = AddConstInput(TensorType_INT32, crops, {2, 2});
    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }
};

// Tests case where block_shape and crops are non-const tensors.
//
// Example usage is as follows:
//    BatchToSpaceNDOpDynamicModel m(input_shape);
//    m.SetInput(input_data);
//    m.SetBlockShape(block_shape);
//    m.SetPaddings(crops);
//    m.Invoke();
class BatchToSpaceNDOpDynamicModel : public BatchToSpaceNDOpModel {
 public:
  BatchToSpaceNDOpDynamicModel(std::initializer_list<int> input_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    block_shape_ = AddInput(TensorType_INT32);
    crops_ = AddInput(TensorType_INT32);
    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({input_shape, {2}, {2, 2}});
  }
};

TEST(BatchToSpaceNDOpTest, SimpleConstTest) {
  BatchToSpaceNDOpConstModel m({4, 2, 2, 1}, {2, 2}, {0, 0, 0, 0});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 5, 2, 6, 9, 13, 10, 14, 3, 7,
                                               4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({4, 2, 2, 1});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 5, 2, 6, 9, 13, 10, 14, 3, 7,
                                               4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, InvalidShapeTest) {
  EXPECT_DEATH(BatchToSpaceNDOpConstModel({3, 2, 2, 1}, {2, 2}, {0, 0, 0, 0}),
               "Cannot allocate tensors");
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
