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
#include "tensorflow/contrib/lite/c/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class GatherOpModel : public SingleOpModel {
 public:
  GatherOpModel(std::initializer_list<int> input_shape, TensorType input_type,
                std::initializer_list<int> positions_shape) {
    input_ = AddInput(input_type);
    positions_ = AddInput(TensorType_INT32);
    output_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_GATHER, BuiltinOptions_GatherOptions,
                 CreateGatherOptions(builder_, 0).Union());
    BuildInterpreter({input_shape, positions_shape});
  }

  void SetInputFloat(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetInputUint8(std::initializer_list<uint8_t> data) {
    PopulateTensor<uint8_t>(input_, data);
  }

  void SetInput(std::initializer_list<string> data) {
    PopulateStringTensor(input_, data);
  }

  void SetPositions(std::initializer_list<int> data) {
    PopulateTensor<int>(positions_, data);
  }

  std::vector<float> GetOutputFloat() { return ExtractVector<float>(output_); }
  std::vector<uint8_t> GetOutputUint8() {
    return ExtractVector<uint8_t>(output_);
  }
  std::vector<string> GetOutputString() {
    return ExtractVector<string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int positions_;
  int output_;
};

TEST(GatherOpTest, Shuffle) {
  GatherOpModel m({2, 2}, TensorType_FLOAT32, {2});
  m.SetInputFloat({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions({1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(),
              ElementsAreArray(ArrayFloatNear({0.7, 0.8, -2, 0.2})));
}

TEST(GatherOpTest, Test0DIndex) {
  GatherOpModel m({2, 2}, TensorType_FLOAT32, {});
  m.SetInputFloat({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(), ElementsAreArray(ArrayFloatNear({0.7, 0.8})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
}

TEST(GatherOpTest, Test0DIndexWith0DResult) {
  // 0D tensor is special case in current TFLite. Test it once to make sure
  // existing workarounds are fine with it.
  GatherOpModel m({3}, TensorType_FLOAT32, {});
  m.SetInputFloat({1.0, 2.0, 3.0});
  m.SetPositions({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(), ElementsAreArray(ArrayFloatNear({2.0})));
  EXPECT_TRUE(m.GetOutputShape().empty());
}

TEST(GatherOpTest, Test2DIndexWith2DResult) {
  GatherOpModel m({3}, TensorType_FLOAT32, {1, 2});
  m.SetInputFloat({1.0, 2.0, 3.0});
  m.SetPositions({1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(), ElementsAreArray(ArrayFloatNear({2.0, 1.0})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(FloatGatherOpTest, Duplicate) {
  GatherOpModel m({1, 2, 2}, TensorType_FLOAT32, {2});
  m.SetInputFloat({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions({0, 0});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutputFloat(),
      ElementsAreArray(ArrayFloatNear({-2, 0.2, 0.7, 0.8, -2, 0.2, 0.7, 0.8})));
}

TEST(FloatGatherOpTest, Slice) {
  GatherOpModel m({4, 1}, TensorType_FLOAT32, {2});
  m.SetInputFloat({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutputFloat(), ElementsAreArray(ArrayFloatNear({0.2, 0.8})));
}

TEST(Uint8tGatherOpTest, Shuffle) {
  GatherOpModel m({2, 2}, TensorType_UINT8, {2});
  m.SetInputUint8({133, 134, 14, 15});
  m.SetPositions({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutputUint8(), ElementsAreArray({14, 15, 133, 134}));
}

TEST(GatherOpTest, SimpleString) {
  GatherOpModel m({3}, TensorType_STRING, {2});
  m.SetInput({"A", "B", "C"});
  m.SetPositions({0, 2});
  m.Invoke();
  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetOutputString(), ElementsAreArray({"A", "C"}));
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
