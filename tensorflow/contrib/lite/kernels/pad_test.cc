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

class PadOpModel : public SingleOpModel {
 public:
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetPaddings(std::initializer_list<int> paddings) {
    PopulateTensor<int>(paddings_, paddings);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int output_;
  int paddings_;
};

// Tests case where paddings is a const tensor.
//
// Example usage is as follows:
//    PadOpDynamicModel m(input_shape, paddings_shape, paddings_data);
//    m.SetInput(input_data);
//    m.Invoke();
class PadOpConstModel : public PadOpModel {
 public:
  PadOpConstModel(std::initializer_list<int> input_shape,
                  std::initializer_list<int> paddings_shape,
                  std::initializer_list<int> paddings) {
    input_ = AddInput(TensorType_FLOAT32);
    paddings_ = AddConstInput(TensorType_INT32, paddings, paddings_shape);
    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                 CreatePadOptions(builder_).Union());
    BuildInterpreter({input_shape});
  }
};

// Test case where paddings is a non-const tensor.
//
// Example usage is as follows:
//    PadOpDynamicModel m(input_shape, paddings_shape);
//    m.SetInput(input_data);
//    m.SetPaddings(paddings_data);
//    m.Invoke();
class PadOpDynamicModel : public PadOpModel {
 public:
  PadOpDynamicModel(std::initializer_list<int> input_shape,
                    std::initializer_list<int> paddings_shape) {
    input_ = AddInput(TensorType_FLOAT32);
    paddings_ = AddInput(TensorType_INT32);
    output_ = AddOutput(TensorType_FLOAT32);

    SetBuiltinOp(BuiltinOperator_PAD, BuiltinOptions_PadOptions,
                 CreatePadOptions(builder_).Union());
    BuildInterpreter({input_shape, paddings_shape});
  }
};

TEST(PadOpTest, TooManyDimensions) {
  EXPECT_DEATH(
      PadOpConstModel({1, 2, 3, 4, 5, 6, 7, 8, 9}, {9, 2},
                      {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9}),
      "dims != 4");
}

TEST(PadOpTest, UnequalDimensions) {
  EXPECT_DEATH(PadOpConstModel({1, 1, 2, 1}, {3, 2}, {1, 1, 2, 2, 3, 3}),
               "3 != 4");
}

TEST(PadOpTest, InvalidPadValue) {
  EXPECT_DEATH(
      PadOpConstModel({1, 1, 2, 1}, {4, 2}, {0, 0, 1, -1, 2, -1, 0, 0}),
      "Pad value has to be greater than equal to 0.");
}

TEST(PadOpTest, SimpleConstTest) {
  // Padding is represented as four 2-D lists representing above padding and
  // below padding (i.e. {{0, 0}, {1, 1}, {1, 1}, {0, 0}}).
  PadOpConstModel m({1, 2, 2, 1}, {4, 2}, {0, 0, 1, 1, 1, 1, 0, 0});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadOpTest, SimpleDynamicTest) {
  PadOpDynamicModel m({1, 2, 2, 1}, {4, 2});
  m.SetInput({1, 2, 3, 4});
  m.SetPaddings({0, 0, 1, 1, 1, 1, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadOpTest, AdvancedConstTest) {
  PadOpConstModel m({1, 2, 3, 1}, {4, 2}, {0, 0, 0, 2, 1, 3, 0, 0});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

TEST(PadOpTest, AdvancedDynamicTest) {
  PadOpDynamicModel m({1, 2, 3, 1}, {4, 2});
  m.SetInput({1, 2, 3, 4, 5, 6});
  m.SetPaddings({0, 0, 0, 2, 1, 3, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(),
              ElementsAreArray({0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 7, 1}));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
