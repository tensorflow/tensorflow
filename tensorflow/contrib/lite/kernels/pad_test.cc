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
  PadOpModel(std::initializer_list<int> input_shape,
             std::initializer_list<int> before_padding,
             std::initializer_list<int> after_padding) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_PAD, BuiltinOptions_PadOptions,
        CreatePadOptions(builder_, builder_.CreateVector<int>(before_padding),
                         builder_.CreateVector<int>(after_padding))
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

TEST(PadOpTest, TooManyDimensions) {
  EXPECT_DEATH(
      PadOpModel({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9},
                 {1, 2, 3, 4, 5, 6, 7, 8, 9}),
      "dims != 4");
}

// TODO(nupurgarg): Test case where before padding and after padding arrays
// don't contain the same number of dimensions.
TEST(PadOpTest, UnequalDimensions) {
  EXPECT_DEATH(PadOpModel({1, 1, 2, 1}, {1, 2, 3}, {1, 2, 3}),
               "dims != op_context.params->num_dimensions");
}

TEST(PadOpTest, InvalidPadValue) {
  EXPECT_DEATH(PadOpModel({1, 1, 2, 1}, {0, 1, 2, 0}, {0, -1, -1, 0}),
               "Pad value has to be greater than equal to 0.");
}

TEST(PadOpTest, SimpleTest) {
  PadOpModel m({1, 2, 2, 1}, {0, 1, 1, 0}, {0, 1, 1, 0});
  m.SetInput({1, 2, 3, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4,
                                               0, 0, 0, 0, 0}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
}

TEST(PadOpTest, AdvancedTest) {
  // The padding is input in the order of batch, height, width, depth.
  PadOpModel m({1, 2, 3, 1}, {0, 0, 1, 0}, {0, 2, 3, 0});
  m.SetInput({1, 2, 3, 4, 5, 6});
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
