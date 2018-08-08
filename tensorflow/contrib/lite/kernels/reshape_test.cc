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
using ::testing::IsEmpty;

class ReshapeOpModel : public SingleOpModel {
 public:
  ReshapeOpModel(std::initializer_list<int> input_shape,
                 std::initializer_list<int> new_shape,
                 bool use_shape_input_tensor = false) {
    input_ = AddInput(TensorType_FLOAT32);
    output_ = AddOutput(TensorType_FLOAT32);
    int shape_input_tensor =
        use_shape_input_tensor ? AddInput(TensorType_INT32) : -1;
    SetBuiltinOp(
        BuiltinOperator_RESHAPE, BuiltinOptions_ReshapeOptions,
        CreateReshapeOptions(builder_, builder_.CreateVector<int>(new_shape))
            .Union());
    if (use_shape_input_tensor) {
      BuildInterpreter({input_shape, GetShape(shape_input_tensor)});
      PopulateTensor<int>(shape_input_tensor, new_shape);
    } else {
      BuildInterpreter({input_shape});
    }
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

TEST(ReshapeOpTest, MismatchedDimensions) {
  EXPECT_DEATH(ReshapeOpModel({1, 2, 4, 1}, {2, 1}),
               "num_input_elements != num_output_elements");
}

TEST(ReshapeOpTest, TooManyDimensions) {
  EXPECT_DEATH(
      ReshapeOpModel({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
      "Found too many dimensions");
}

TEST(ReshapeOpTest, TooManySpecialDimensions) {
  EXPECT_DEATH(ReshapeOpModel({1, 2, 4, 1}, {-1, -1, 2, 4}),
               "stretch_dim != -1");
}

TEST(ReshapeOpTest, SimpleTest) {
  ReshapeOpModel m({1, 2, 4, 1}, {2, 2, 2});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST(ReshapeOpTest, ShapeTensorInput) {
  ReshapeOpModel m({1, 2, 4, 1}, {2, 2, 2}, /*use_shape_input_tensor=*/true);
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 2, 2}));
}

TEST(ReshapeOpTest, WithStretchDimension) {
  ReshapeOpModel m({1, 2, 4, 1}, {2, 1, -1});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 1, 4}));
}

TEST(ReshapeOpTest, ScalarOutput) {
  ReshapeOpModel m({1}, {});
  m.SetInput({3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
}

TEST(ReshapeOpTest, LegacyScalarOutput) {
  ReshapeOpModel m({1}, {0});
  m.SetInput({3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
