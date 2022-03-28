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

#include <initializer_list>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class OneHotOpModel : public SingleOpModel {
 public:
  OneHotOpModel(std::initializer_list<int> input_shape, int depth_value,
                TensorType dtype, int axis = -1, T on_value = 1,
                T off_value = 0, TensorType indices_type = TensorType_INT32) {
    indices_ = AddInput(indices_type);
    int depth = AddInput(TensorType_INT32);
    int on = AddInput(dtype);
    int off = AddInput(dtype);
    output_ = AddOutput(dtype);
    SetBuiltinOp(BuiltinOperator_ONE_HOT, BuiltinOptions_OneHotOptions,
                 CreateOneHotOptions(builder_, axis).Union());
    BuildInterpreter({input_shape});

    PopulateTensor<int>(depth, {depth_value});
    PopulateTensor<T>(on, {on_value});
    PopulateTensor<T>(off, {off_value});
  }

  template <typename TI>
  void SetIndices(std::initializer_list<TI> data) {
    PopulateTensor<TI>(indices_, data);
  }

  TfLiteStatus InvokeWithResult() { return interpreter_->Invoke(); }

  int32_t GetOutputSize() { return GetTensorSize(output_); }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int indices_;
  int output_;
};

TEST(OneHotOpTest, BasicFloat) {
  const int depth = 3;
  OneHotOpModel<float> model({3}, depth, TensorType_FLOAT32);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}));
}

TEST(OneHotOpTest, BasicInt) {
  const int depth = 3;
  OneHotOpModel<int> model({3}, depth, TensorType_INT32);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST(OneHotOpTest, BasicInt8) {
  const int depth = 3;
  OneHotOpModel<int8_t> model({3}, depth, TensorType_INT8);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST(OneHotOpTest, BasicUint8) {
  const int depth = 3;
  OneHotOpModel<uint8_t> model({3}, depth, TensorType_UINT8);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

TEST(OneHotOpTest, BasicBool) {
  const int depth = 3;
  OneHotOpModel<bool> model({3}, depth, TensorType_BOOL);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({true, false, false, false, true, false, false,
                                false, true}));
}

TEST(OneHotOpTest, SmallDepth) {
  const int depth = 1;
  OneHotOpModel<int> model({3}, depth, TensorType_INT32);
  model.SetIndices({0, 1, 2});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 1}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0}));
}

TEST(OneHotOpTest, BigDepth) {
  const int depth = 4;
  OneHotOpModel<int> model({2}, depth, TensorType_INT32);
  model.SetIndices({0, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 4}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 0, 1, 0, 0}));
}

TEST(OneHotOpTest, OnOffValues) {
  const int depth = 3;
  const int axis = -1;
  const int on = 5;
  const int off = 0;
  OneHotOpModel<int> model({4}, depth, TensorType_INT32, axis, on, off);
  model.SetIndices({0, 2, -1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({4, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0}));
}

TEST(OneHotOpTest, ZeroAxis) {
  const int depth = 3;
  const int axis = 0;
  const int on = 5;
  const int off = 0;
  OneHotOpModel<int> model({4}, depth, TensorType_INT32, axis, on, off);
  model.SetIndices({0, 2, -1, 1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 4}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({5, 0, 0, 0, 0, 0, 0, 5, 0, 5, 0, 0}));
}

TEST(OneHotOpTest, MultiDimensionalIndices) {
  const int depth = 3;
  const int axis = -1;
  const float on = 2;
  const float off = 0;
  OneHotOpModel<float> model({2, 2}, depth, TensorType_FLOAT32, axis, on, off);
  model.SetIndices({0, 2, 1, -1});
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 3}));
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0}));
}

TEST(OneHotOpTest, Int64Indices) {
  const int depth = 3;
  const int axis = -1;
  const int on = 1;
  const int off = 0;
  OneHotOpModel<int> model({3}, depth, TensorType_INT32, axis, on, off,
                           TensorType_INT64);
  std::initializer_list<int64_t> indices = {0, 1, 2};
  model.SetIndices(indices);
  ASSERT_EQ(model.InvokeUnchecked(), kTfLiteOk);

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 3}));
  EXPECT_THAT(model.GetOutput(), ElementsAreArray({1, 0, 0, 0, 1, 0, 0, 0, 1}));
}

}  // namespace
}  // namespace tflite
