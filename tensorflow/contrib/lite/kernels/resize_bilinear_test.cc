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

class ResizeBilinearOpModel : public SingleOpModel {
 public:
  ResizeBilinearOpModel(std::initializer_list<int> input_shape, int new_height,
                        int new_width) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    SetBuiltinOp(
        BuiltinOperator_RESIZE_BILINEAR, BuiltinOptions_ResizeBilinearOptions,
        CreateResizeBilinearOptions(builder_, new_height, new_width).Union());
    BuildInterpreter({input_shape});
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int output_;
};

TEST(ResizeBilinearOpTest, HorizontalResize) {
  ResizeBilinearOpModel m({1, 1, 2, 1}, 1, 3);
  m.SetInput({3, 6});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 5, 6})));
}

TEST(ResizeBilinearOpTest, VerticalResize) {
  ResizeBilinearOpModel m({1, 2, 1, 1}, 3, 1);
  m.SetInput({3, 9});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 7, 9})));
}

TEST(ResizeBilinearOpTest, TwoDimensionalResize) {
  ResizeBilinearOpModel m({1, 2, 2, 1}, 3, 3);
  m.SetInput({
      3, 6,  //
      9, 12  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 3, 5, 6,    //
                                 7, 9, 10,   //
                                 9, 11, 12,  //
                             })));
}

TEST(ResizeBilinearOpTest, TwoDimensionalResizeWithTwoBatches) {
  ResizeBilinearOpModel m({2, 2, 2, 1}, 3, 3);
  m.SetInput({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 3, 5, 6,     //
                                 7, 9, 10,    //
                                 9, 11, 12,   //
                                 4, 8, 10,    //
                                 8, 12, 14,   //
                                 10, 14, 16,  //
                             })));
}

TEST(ResizeBilinearOpTest, ThreeDimensionalResize) {
  ResizeBilinearOpModel m({1, 2, 2, 2}, 3, 3);
  m.SetInput({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 3, 4, 5, 8, 6, 10,      //
                                 7, 8, 9, 12, 10, 14,    //
                                 9, 10, 11, 14, 12, 16,  //
                             })));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  // On Linux, add: tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
