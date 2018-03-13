
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
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class TopKV2OpModel : public SingleOpModel {
 public:
  TopKV2OpModel(std::initializer_list<int> input_shape, TensorType input_type,
                int top_k) {
    input_ = AddInput(input_type);
    top_k_ = AddInput(TensorType_INT32);
    output_indexes_ = AddOutput(TensorType_INT32);
    output_values_ = AddOutput(input_type);
    SetBuiltinOp(BuiltinOperator_TOPK_V2, BuiltinOptions_TopKV2Options, 0);
    BuildInterpreter({input_shape, {1}});
    PopulateTensor<int32_t>(top_k_, {top_k});
  }

  void SetInputFloat(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetInputUInt8(std::initializer_list<uint8> data) {
    PopulateTensor<uint8>(input_, data);
  }

  void SetInputInt32(std::initializer_list<int32> data) {
    PopulateTensor<int32>(input_, data);
  }

  void SetInputInt64(std::initializer_list<int64_t> data) {
    PopulateTensor<int64_t>(input_, data);
  }

  std::vector<int32> GetIndexes() {
    return ExtractVector<int32>(output_indexes_);
  }

  std::vector<float> GetValuesFloat() {
    return ExtractVector<float>(output_values_);
  }

  std::vector<uint8> GetValuesUInt8() {
    return ExtractVector<uint8>(output_values_);
  }

  std::vector<int32> GetValuesInt32() {
    return ExtractVector<int32>(output_values_);
  }

  std::vector<int64_t> GetValuesInt64() {
    return ExtractVector<int64_t>(output_values_);
  }

 protected:
  int input_;
  int top_k_;
  int output_indexes_;
  int output_values_;
};

// The test where the tensor dimension is equal to top.
TEST(TopKV2OpTest, EqualFloat) {
  TopKV2OpModel m({2, 2}, TensorType_FLOAT32, 2);
  m.SetInputFloat({-2.0, 0.2, 0.8, 0.1});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({1, 0, 0, 1}));
  EXPECT_THAT(m.GetValuesFloat(),
              ElementsAreArray(ArrayFloatNear({0.2, -2.0, 0.8, 0.1})));
}

// Test when internal dimension is k+1.
TEST(TopKV2OpTest, BorderFloat) {
  TopKV2OpModel m({2, 3}, TensorType_FLOAT32, 2);
  m.SetInputFloat({-2.0, -3.0, 0.2, 0.8, 0.1, -0.1});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 0, 0, 1}));
  EXPECT_THAT(m.GetValuesFloat(),
              ElementsAreArray(ArrayFloatNear({0.2, -2.0, 0.8, 0.1})));
}
// Test when internal dimension is higher than k.
TEST(TopKV2OpTest, LargeFloat) {
  TopKV2OpModel m({2, 4}, TensorType_FLOAT32, 2);
  m.SetInputFloat({-2.0, -3.0, -4.0, 0.2, 0.8, 0.1, -0.1, -0.8});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({3, 0, 0, 1}));
  EXPECT_THAT(m.GetValuesFloat(),
              ElementsAreArray(ArrayFloatNear({0.2, -2.0, 0.8, 0.1})));
}

// Test 1D case.
TEST(TopKV2OpTest, VectorFloat) {
  TopKV2OpModel m({8}, TensorType_FLOAT32, 2);
  m.SetInputFloat({-2.0, -3.0, -4.0, 0.2, 0.8, 0.1, -0.1, -0.8});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({4, 3}));
  EXPECT_THAT(m.GetValuesFloat(), ElementsAreArray(ArrayFloatNear({0.8, 0.2})));
}

// Check that uint8 works.
TEST(TopKV2OpTest, TypeUint8) {
  TopKV2OpModel m({2, 3}, TensorType_UINT8, 2);
  m.SetInputUInt8({1, 2, 3, 251, 250, 249});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 0, 1}));
  EXPECT_THAT(m.GetValuesUInt8(), ElementsAreArray({3, 2, 251, 250}));
}

// Check that int32 works.
TEST(TopKV2OpTest, TypeInt32) {
  TopKV2OpModel m({2, 3}, TensorType_INT32, 2);
  m.SetInputInt32({1, 2, 3, 10251, 10250, 10249});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 0, 1}));
  EXPECT_THAT(m.GetValuesInt32(), ElementsAreArray({3, 2, 10251, 10250}));
}

// Check that int64 works.
TEST(TopKV2OpTest, TypeInt64) {
  TopKV2OpModel m({2, 3}, TensorType_INT64, 2);
  m.SetInputInt64({1, 2, 3, -1, -2, -3});
  m.Invoke();
  EXPECT_THAT(m.GetIndexes(), ElementsAreArray({2, 1, 0, 1}));
  EXPECT_THAT(m.GetValuesInt64(), ElementsAreArray({3, 2, -1, -2}));
}
}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
