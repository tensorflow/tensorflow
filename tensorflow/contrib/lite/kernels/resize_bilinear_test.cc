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
  ResizeBilinearOpModel(const TensorData& input,
                        std::initializer_list<int> size_data = {}) {
    bool const_size = size_data.size() != 0;
    input_ = AddInput(input);
    if (const_size) {
      size_ = AddConstInput(TensorType_INT32, size_data, {2});
    } else {
      size_ = AddInput({TensorType_INT32, {2}});
    }
    output_ = AddOutput(TensorType_FLOAT32);  // Always float.
    SetBuiltinOp(BuiltinOperator_RESIZE_BILINEAR,
                 BuiltinOptions_ResizeBilinearOptions,
                 CreateResizeBilinearOptions(builder_).Union());
    if (const_size) {
      BuildInterpreter({GetShape(input_)});
    } else {
      BuildInterpreter({GetShape(input_), GetShape(size_)});
    }
  }

  void SetInput(std::initializer_list<float> data) {
    PopulateTensor(input_, data);
  }
  void SetSize(std::initializer_list<int> data) { PopulateTensor(size_, data); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

 private:
  int input_;
  int size_;
  int output_;
};

TEST(ResizeBilinearOpTest, HorizontalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 1, 2, 1}});
  m.SetInput({3, 6});
  m.SetSize({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 5, 6})));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 1, 2, 1}}, {1, 3});
  const_m.SetInput({3, 6});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 5, 6})));
}

TEST(ResizeBilinearOpTest, VerticalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 1, 1}});
  m.SetInput({3, 9});
  m.SetSize({3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 7, 9})));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 2, 1, 1}}, {3, 1});
  const_m.SetInput({3, 9});
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput(), ElementsAreArray(ArrayFloatNear({3, 7, 9})));
}

TEST(ResizeBilinearOpTest, TwoDimensionalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 2, 1}});
  m.SetInput({
      3, 6,  //
      9, 12  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 3, 5, 6,    //
                                 7, 9, 10,   //
                                 9, 11, 12,  //
                             })));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 2, 2, 1}}, {3, 3});
  const_m.SetInput({
      3, 6,  //
      9, 12  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                       3, 5, 6,    //
                                       7, 9, 10,   //
                                       9, 11, 12,  //
                                   })));
}

TEST(ResizeBilinearOpTest, TwoDimensionalResizeWithTwoBatches) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {2, 2, 2, 1}});
  m.SetInput({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 3, 5, 6,     //
                                 7, 9, 10,    //
                                 9, 11, 12,   //
                                 4, 8, 10,    //
                                 8, 12, 14,   //
                                 10, 14, 16,  //
                             })));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {2, 2, 2, 1}}, {3, 3});
  const_m.SetInput({
      3, 6,   //
      9, 12,  //
      4, 10,  //
      10, 16  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                       3, 5, 6,     //
                                       7, 9, 10,    //
                                       9, 11, 12,   //
                                       4, 8, 10,    //
                                       8, 12, 14,   //
                                       10, 14, 16,  //
                                   })));
}

TEST(ResizeBilinearOpTest, ThreeDimensionalResize) {
  ResizeBilinearOpModel m({TensorType_FLOAT32, {1, 2, 2, 2}});
  m.SetInput({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  m.SetSize({3, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                 3, 4, 5, 8, 6, 10,      //
                                 7, 8, 9, 12, 10, 14,    //
                                 9, 10, 11, 14, 12, 16,  //
                             })));

  ResizeBilinearOpModel const_m({TensorType_FLOAT32, {1, 2, 2, 2}}, {3, 3});
  const_m.SetInput({
      3, 4, 6, 10,    //
      9, 10, 12, 16,  //
  });
  const_m.Invoke();
  EXPECT_THAT(const_m.GetOutput(), ElementsAreArray(ArrayFloatNear({
                                       3, 4, 5, 8, 6, 10,      //
                                       7, 8, 9, 12, 10, 14,    //
                                       9, 10, 11, 14, 12, 16,  //
                                   })));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
