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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::IsEmpty;

class BaseSqueezeOpModel : public SingleOpModel {
 public:
  BaseSqueezeOpModel(const TensorData& input, const TensorData& output,
                     std::initializer_list<int> axis) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    SetBuiltinOp(
        BuiltinOperator_SQUEEZE, BuiltinOptions_SqueezeOptions,
        CreateSqueezeOptions(builder_, builder_.CreateVector<int>(axis))
            .Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }

 protected:
  int input_;
  int output_;
};

template <typename T>
class SqueezeOpModel : public BaseSqueezeOpModel {
 public:
  using BaseSqueezeOpModel::BaseSqueezeOpModel;

  void SetInput(std::initializer_list<T> data) { PopulateTensor(input_, data); }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }
};

template <typename T>
class SqueezeOpTest : public ::testing::Test {};

using DataTypes = ::testing::Types<float, int8_t, int16_t, int32_t>;
TYPED_TEST_SUITE(SqueezeOpTest, DataTypes);

TYPED_TEST(SqueezeOpTest, SqueezeAll) {
  std::initializer_list<TypeParam> data = {1,  2,  3,  4,  5,  6,  7,  8,
                                           9,  10, 11, 12, 13, 14, 15, 16,
                                           17, 18, 19, 20, 21, 22, 23, 24};
  SqueezeOpModel<TypeParam> m({GetTensorType<TypeParam>(), {1, 24, 1}},
                              {GetTensorType<TypeParam>(), {24}}, {});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}));
}

TYPED_TEST(SqueezeOpTest, SqueezeSelectedAxis) {
  std::initializer_list<TypeParam> data = {1,  2,  3,  4,  5,  6,  7,  8,
                                           9,  10, 11, 12, 13, 14, 15, 16,
                                           17, 18, 19, 20, 21, 22, 23, 24};
  SqueezeOpModel<TypeParam> m({GetTensorType<TypeParam>(), {1, 24, 1}},
                              {GetTensorType<TypeParam>(), {24}}, {2});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}));
}

TYPED_TEST(SqueezeOpTest, SqueezeNegativeAxis) {
  std::initializer_list<TypeParam> data = {1,  2,  3,  4,  5,  6,  7,  8,
                                           9,  10, 11, 12, 13, 14, 15, 16,
                                           17, 18, 19, 20, 21, 22, 23, 24};
  SqueezeOpModel<TypeParam> m({GetTensorType<TypeParam>(), {1, 24, 1}},
                              {GetTensorType<TypeParam>(), {24}}, {-1, 0});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({24}));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray({1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}));
}

TYPED_TEST(SqueezeOpTest, SqueezeAllDims) {
  std::initializer_list<TypeParam> data = {3};
  SqueezeOpModel<TypeParam> m(
      {GetTensorType<TypeParam>(), {1, 1, 1, 1, 1, 1, 1}},
      {GetTensorType<TypeParam>(), {1}}, {});
  m.SetInput(data);
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), IsEmpty());
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({3}));
}

}  // namespace
}  // namespace tflite
