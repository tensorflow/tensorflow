/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

class SelectOpModel : public SingleOpModel {
 public:
  SelectOpModel(std::initializer_list<int> input1_shape,
                std::initializer_list<int> input2_shape,
                std::initializer_list<int> input3_shape,
                TensorType input_type) {
    input1_ = AddInput(TensorType_BOOL);
    input2_ = AddInput(input_type);
    input3_ = AddInput(input_type);
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_SELECT, BuiltinOptions_SelectOptions,
                 CreateSelectOptions(builder_).Union());
    BuildInterpreter({input1_shape, input2_shape, input3_shape});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }
  int input3() { return input3_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input1_;
  int input2_;
  int input3_;
  int output_;
};

TEST(SelectOpTest, SelectBool) {
  SelectOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, {1, 1, 1, 4},
                      TensorType_BOOL);

  model.PopulateTensor<bool>(model.input1(), {true, false, true, false});
  model.PopulateTensor<bool>(model.input2(), {false, false, false, false});
  model.PopulateTensor<bool>(model.input3(), {true, true, true, true});
  model.Invoke();

  EXPECT_THAT(model.GetOutput<bool>(),
              ElementsAreArray({false, true, false, true}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SelectOpTest, SelectFloat) {
  SelectOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, {1, 1, 1, 4},
                      TensorType_FLOAT32);

  model.PopulateTensor<bool>(model.input1(), {true, false, true, false});
  model.PopulateTensor<float>(model.input2(), {0.1, 0.2, 0.3, 0.4});
  model.PopulateTensor<float>(model.input3(), {0.5, 0.6, 0.7, 0.8});
  model.Invoke();

  EXPECT_THAT(model.GetOutput<float>(), ElementsAreArray({0.1, 0.6, 0.3, 0.8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SelectOpTest, SelectUInt8) {
  SelectOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, {1, 1, 1, 4},
                      TensorType_UINT8);

  model.PopulateTensor<bool>(model.input1(), {false, true, false, false});
  model.PopulateTensor<uint8>(model.input2(), {1, 2, 3, 4});
  model.PopulateTensor<uint8>(model.input3(), {5, 6, 7, 8});
  model.Invoke();

  EXPECT_THAT(model.GetOutput<uint8>(), ElementsAreArray({5, 2, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SelectOpTest, SelectInt32) {
  SelectOpModel model({1, 1, 1, 4}, {1, 1, 1, 4}, {1, 1, 1, 4},
                      TensorType_INT32);

  model.PopulateTensor<bool>(model.input1(), {false, true, false, false});
  model.PopulateTensor<int32>(model.input2(), {1, 2, 3, 4});
  model.PopulateTensor<int32>(model.input3(), {5, 6, 7, 8});
  model.Invoke();

  EXPECT_THAT(model.GetOutput<int32>(), ElementsAreArray({5, 2, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 1, 1, 4}));
}

TEST(SelectOpTest, RankOneSelectInt32) {
  SelectOpModel model({2}, {2, 1, 2, 1}, {2, 1, 2, 1}, TensorType_INT32);

  model.PopulateTensor<bool>(model.input1(), {false, true});
  model.PopulateTensor<int32>(model.input2(), {1, 2, 3, 4});
  model.PopulateTensor<int32>(model.input3(), {5, 6, 7, 8});
  model.Invoke();

  EXPECT_THAT(model.GetOutput<int32>(), ElementsAreArray({5, 6, 3, 4}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 1, 2, 1}));
}

TEST(SelectOpTest, RankZeroSelectInt32) {
  SelectOpModel model({1}, {1, 2, 2, 1}, {1, 2, 2, 1}, TensorType_INT32);

  model.PopulateTensor<bool>(model.input1(), {false});
  model.PopulateTensor<int32>(model.input2(), {1, 2, 3, 4});
  model.PopulateTensor<int32>(model.input3(), {5, 6, 7, 8});
  model.Invoke();

  EXPECT_THAT(model.GetOutput<int32>(), ElementsAreArray({5, 6, 7, 8}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
