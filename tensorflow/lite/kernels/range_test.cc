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
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, DeltaGreaterThanOne) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {2});
  model.PopulateTensor<int32_t>(model.limit(), {9});
  model.PopulateTensor<int32_t>(model.delta(), {2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, NegativeDelta) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {10});
  model.PopulateTensor<int32_t>(model.limit(), {3});
  model.PopulateTensor<int32_t>(model.delta(), {-3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, FloatSimple) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {0});
  model.PopulateTensor<float>(model.limit(), {4});
  model.PopulateTensor<float>(model.delta(), {1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(0, 1, 2, 3));
}

TEST(RangeOpModel, FloatDeltaGreaterThanOne) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {2});
  model.PopulateTensor<float>(model.limit(), {9});
  model.PopulateTensor<float>(model.delta(), {2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(4));
  EXPECT_THAT(model.GetOutput(), ElementsAre(2, 4, 6, 8));
}

TEST(RangeOpModel, FloatNegativeDelta) {
  RangeOpModel<float> model(TensorType_FLOAT32);
  model.PopulateTensor<float>(model.start(), {10});
  model.PopulateTensor<float>(model.limit(), {3});
  model.PopulateTensor<float>(model.delta(), {-3});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(3));
  EXPECT_THAT(model.GetOutput(), ElementsAre(10, 7, 4));
}

TEST(RangeOpModel, EmptyOutput) {
  RangeOpModel<int32_t> model(TensorType_INT32);
  model.PopulateTensor<int32_t>(model.start(), {0});
  model.PopulateTensor<int32_t>(model.limit(), {0});
  model.PopulateTensor<int32_t>(model.delta(), {1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);
  EXPECT_THAT(model.GetOutputShape(), ElementsAre(0));
  EXPECT_THAT(model.GetOutput(), ElementsAre());
}

}  // namespace
}  // namespace tflite
