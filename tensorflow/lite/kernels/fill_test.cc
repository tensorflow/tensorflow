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

class FillOpModel : public SingleOpModel {
 public:
  explicit FillOpModel(const TensorData& input1, const TensorData& input2) {
    input1_ = AddInput(input1);
    input2_ = AddInput(input2);
    output_ = AddOutput(input1);
    SetBuiltinOp(BuiltinOperator_FILL, BuiltinOptions_FillOptions,
                 CreateFillOptions(builder_).Union());
    BuildInterpreter({GetShape(input1_), GetShape(input2_)});
  }

  int input1() { return input1_; }
  int input2() { return input2_; }
  int output() { return output_; }

 protected:
  int input1_;
  int input2_;
  int output_;
};

TEST(FillOpModel, FillInt32) {
  FillOpModel m({TensorType_INT32, {2}}, {TensorType_INT32});
  m.PopulateTensor<int32_t>(m.input1(), {2, 3});
  m.PopulateTensor<int32_t>(m.input2(), {-11});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({-11, -11, -11, -11, -11, -11}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 3}));
}

TEST(FillOpModel, FillInt64) {
  FillOpModel m({TensorType_INT32, {2}}, {TensorType_INT64});
  m.PopulateTensor<int32_t>(m.input1(), {2, 4});
  m.PopulateTensor<int64_t>(m.input2(), {1LL << 45});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<int64_t>(m.output()),
              ElementsAreArray({1LL << 45, 1LL << 45, 1LL << 45, 1LL << 45,
                                1LL << 45, 1LL << 45, 1LL << 45, 1LL << 45}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 4}));
}

TEST(FillOpModel, FillFloat) {
  FillOpModel m({TensorType_INT64, {3}}, {TensorType_FLOAT32});
  m.PopulateTensor<int64_t>(m.input1(), {2, 2, 2});
  m.PopulateTensor<float>(m.input2(), {4.0});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              ElementsAreArray({4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 2, 2}));
}

TEST(FillOpModel, FillOutputScalar) {
  FillOpModel m({TensorType_INT64, {0}}, {TensorType_FLOAT32});
  m.PopulateTensor<float>(m.input2(), {4.0});
  m.Invoke();
  EXPECT_THAT(m.ExtractVector<float>(m.output()), ElementsAreArray({4.0}));
  EXPECT_THAT(m.GetTensorShape(m.output()), IsEmpty());
}

TEST(FillOpModel, FillBool) {
  FillOpModel m({TensorType_INT64, {3}}, {TensorType_BOOL});
  m.PopulateTensor<int64_t>(m.input1(), {2, 2, 2});
  m.PopulateTensor<bool>(m.input2(), {true});
  m.Invoke();
  EXPECT_THAT(
      m.ExtractVector<bool>(m.output()),
      ElementsAreArray({true, true, true, true, true, true, true, true}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 2, 2}));
}

TEST(FillOpModel, FillString) {
  FillOpModel m({TensorType_INT64, {3}}, {TensorType_STRING});
  m.PopulateTensor<int64_t>(m.input1(), {2, 2, 2});
  m.PopulateTensor<std::string>(m.input2(), {"AB"});
  m.Invoke();
  EXPECT_THAT(
      m.ExtractVector<std::string>(m.output()),
      ElementsAreArray({"AB", "AB", "AB", "AB", "AB", "AB", "AB", "AB"}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 2, 2}));
}

}  // namespace
}  // namespace tflite
