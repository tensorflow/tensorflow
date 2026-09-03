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

#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAre;

template <typename T>
class RangeOpModel : public SingleOpModel {
 public:
  explicit RangeOpModel(const TensorType& dtype) {
    start_ = AddInput(dtype);
    limit_ = AddInput(dtype);
    delta_ = AddInput(dtype);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  explicit RangeOpModel(const TensorType& dtype, const std::vector<T>& start,
                        const std::vector<T>& limit,
                        const std::vector<T>& delta) {
    start_ = AddConstInput(dtype, start);
    limit_ = AddConstInput(dtype, limit);
    delta_ = AddConstInput(dtype, delta);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_RANGE, BuiltinOptions_RangeOptions,
                 CreateRangeOptions(builder_).Union());
    BuildInterpreter({GetShape(start_), GetShape(limit_), GetShape(delta_)});
  }

  int start() { return start_; }
  int limit() { return limit_; }
  int delta() { return delta_; }

  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int start_;
  int limit_;
  int delta_;
  int output_;
};

TEST(RangeOpModel, Simple) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {0});
  model.PopulateTensor<int32_t>(model.limit(), {4});
  model.PopulateTensor<int32_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, SimpleConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {0}, {4}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, DeltaGreaterThanOne) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {2});
  model.PopulateTensor<int32_t>(model.limit(), {9});
  model.PopulateTensor<int32_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, DeltaGreaterThanOneConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, NegativeDelta) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {10});
  model.PopulateTensor<int32_t>(model.limit(), {3});
  model.PopulateTensor<int32_t>(model.delta(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, NegativeDeltaConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {10}, {3}, {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatSimple) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {0});
  model.PopulateTensor<float>(model.limit(), {4});
  model.PopulateTensor<float>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, FloatSimpleConst) {
  RangeOpModel<float> model(TensorType_FLOAT32, {0}, {4}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, FloatDeltaGreaterThanOne) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {2});
  model.PopulateTensor<float>(model.limit(), {9});
  model.PopulateTensor<float>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, FloatDeltaGreaterThanOneConst) {
  RangeOpModel<float> model(TensorType_FLOAT32, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, FloatNegativeDelta) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {10});
  model.PopulateTensor<float>(model.limit(), {3});
  model.PopulateTensor<float>(model.delta(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatNegativeDeltaConst) {
  RangeOpModel<float> model(TensorType_FLOAT32, {10}, {3}, {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatRejectsOutputSizeLargerThanIntMax) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {0});
  model.PopulateTensor<float>(
      model.limit(),
      {static_cast<float>(std::numeric_limits<int>::max()) * 2.0f});
  model.PopulateTensor<float>(model.delta(), {1});
  EXPECT_EQ(model.Invoke(), kTfLiteError);
}

TEST(RangeOpModel, EmptyOutput) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {0});
  model.PopulateTensor<int32_t>(model.limit(), {0});
  model.PopulateTensor<int32_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, EmptyOutputConst) {
  RangeOpModel<int32_t> model(TensorType_INT32, {0}, {0}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, Int64Simple) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {0});
  model.PopulateTensor<int64_t>(model.limit(), {4});
  model.PopulateTensor<int64_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, Int64SimpleConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {0}, {4}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, Int64DeltaGreaterThanOne) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {2});
  model.PopulateTensor<int64_t>(model.limit(), {9});
  model.PopulateTensor<int64_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int64DeltaGreaterThanOneConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {2}, {9}, {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, Int64NegativeDelta) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {10});
  model.PopulateTensor<int64_t>(model.limit(), {3});
  model.PopulateTensor<int64_t>(model.delta(), {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, Int64NegativeDeltaConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {10}, {3}, {-3});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, Int64RejectsSubtractionOverflowOutputSize) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(),
                                {std::numeric_limits<int64_t>::min()});
  model.PopulateTensor<int64_t>(model.limit(),
                                {std::numeric_limits<int64_t>::max()});
  model.PopulateTensor<int64_t>(model.delta(), {1});
  EXPECT_EQ(model.Invoke(), kTfLiteError);
}

TEST(RangeOpModel, Int64RejectsOutputSizeLargerThanIntMax) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {0});
  model.PopulateTensor<int64_t>(
      model.limit(),
      {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1});
  model.PopulateTensor<int64_t>(model.delta(), {1});
  EXPECT_EQ(model.Invoke(), kTfLiteError);
}

TEST(RangeOpModel, Int64NoOverflowAfterLastPositiveValue) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(
      model.start(), {std::numeric_limits<int64_t>::max() - 1});
  model.PopulateTensor<int64_t>(model.limit(),
                                {std::numeric_limits<int64_t>::max()});
  model.PopulateTensor<int64_t>(model.delta(), {2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(std::numeric_limits<int64_t>::max() - 1));
}

TEST(RangeOpModel, Int64NoOverflowAfterLastNegativeValue) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(
      model.start(), {std::numeric_limits<int64_t>::min() + 1});
  model.PopulateTensor<int64_t>(model.limit(),
                                {std::numeric_limits<int64_t>::min()});
  model.PopulateTensor<int64_t>(model.delta(), {-2});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(1));
  EXPECT_THAT(model.GetOutput(),
              ElementsAre(std::numeric_limits<int64_t>::min() + 1));
}

TEST(RangeOpModel, Int64EmptyOutput) {
  RangeOpModel<int64_t> model(TensorType_INT64);
  model.PopulateTensor<int64_t>(model.start(), {0});
  model.PopulateTensor<int64_t>(model.limit(), {0});
  model.PopulateTensor<int64_t>(model.delta(), {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

TEST(RangeOpModel, Int64EmptyOutputConst) {
  RangeOpModel<int64_t> model(TensorType_INT64, {0}, {0}, {1});
  ASSERT_EQ(model.Invoke(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

}  // namespace
}  // namespace tflite
