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
#include <stdint.h>

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class ZerosLikeOpModel : public SingleOpModel {
 public:
  explicit ZerosLikeOpModel(const TensorData& input) {
    input_ = AddInput(input);
    output_ = AddOutput(input);
    SetBuiltinOp(BuiltinOperator_ZEROS_LIKE, BuiltinOptions_ZerosLikeOptions,
                 CreateZerosLikeOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  int output() { return output_; }

 protected:
  int input_;
  int output_;
};

TEST(ZerosLikeOpModel, ZerosLikeFloat) {
  ZerosLikeOpModel m({TensorType_FLOAT32, {2, 3}});
  m.PopulateTensor<float>(m.input(), {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<float>(m.output()),
              Pointwise(FloatingPointEq(), {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({2, 3}));
}

TEST(ZerosLikeOpModel, ZerosLikeInt32) {
  ZerosLikeOpModel m({TensorType_INT32, {1, 2, 2, 1}});
  m.PopulateTensor<int32_t>(m.input(), {-2, -1, 0, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int32_t>(m.output()),
              ElementsAreArray({0, 0, 0, 0}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 2, 2, 1}));
}

TEST(ZerosLikeOpModel, ZerosLikeInt64) {
  ZerosLikeOpModel m({TensorType_INT64, {1, 2, 2, 1}});
  m.PopulateTensor<int64_t>(m.input(), {-2, -1, 0, 3});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.ExtractVector<int64_t>(m.output()),
              ElementsAreArray({0, 0, 0, 0}));
  EXPECT_THAT(m.GetTensorShape(m.output()), ElementsAreArray({1, 2, 2, 1}));
}

TEST(ZerosLikeOpModel, InvalidTypeTest) {
  ZerosLikeOpModel m_uint8({TensorType_UINT8, {1, 1}});
  ASSERT_NE(m_uint8.Invoke(), kTfLiteOk)
      << "ZerosLike only currently supports int64, int32, and float32";
  ZerosLikeOpModel m_int16({TensorType_INT16, {1, 1}});
  ASSERT_NE(m_int16.Invoke(), kTfLiteOk)
      << "ZerosLike only currently supports int64, int32, and float32";
  ZerosLikeOpModel m_complex({TensorType_COMPLEX64, {1, 1}});
  ASSERT_NE(m_complex.Invoke(), kTfLiteOk)
      << "ZerosLike only currently supports int64, int32, and float32";
  ZerosLikeOpModel m_int8({TensorType_INT8, {1, 1}});
  ASSERT_NE(m_int8.Invoke(), kTfLiteOk)
      << "ZerosLike only currently supports int64, int32, and float32";
}

}  // namespace
}  // namespace tflite
