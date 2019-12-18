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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class GatherOpModel : public SingleOpModel {
 public:
  GatherOpModel(const TensorData& input, const TensorData& positions,
                int axis = 0) {
    input_ = AddInput(input);
    positions_ = AddInput(positions);
    output_ = AddOutput(input.type);
    SetBuiltinOp(BuiltinOperator_GATHER, BuiltinOptions_GatherOptions,
                 CreateGatherOptions(builder_, axis).Union());
    BuildInterpreter({GetShape(input_), GetShape(positions_)});
  }

  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  void SetStringInput(std::initializer_list<string> data) {
    PopulateStringTensor(input_, data);
  }

  template <typename T>
  void SetPositions(std::initializer_list<T> data) {
    PopulateTensor<T>(positions_, data);
  }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  std::vector<string> GetStringOutput() {
    return ExtractVector<string>(output_);
  }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int positions_;
  int output_;
};

TEST(GatherOpTest, Shuffle) {
  GatherOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<float>({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({0.7, 0.8, -2, 0.2})));
}

TEST(GatherOpTest, Test0DIndex) {
  GatherOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {}});
  m.SetInput<float>({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions<int32_t>({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({0.7, 0.8})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
}

TEST(GatherOpTest, Test0DIndexWith0DResult) {
  // 0D tensor is special case in current TFLite. Test it once to make sure
  // existing workarounds are fine with it.
  GatherOpModel m({TensorType_FLOAT32, {3}}, {TensorType_INT32, {}});
  m.SetInput<float>({1.0, 2.0, 3.0});
  m.SetPositions<int32_t>({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({2.0})));
  EXPECT_TRUE(m.GetOutputShape().empty());
}

TEST(GatherOpTest, Test1DInput1DIndex) {
  GatherOpModel m({TensorType_FLOAT32, {3}}, {TensorType_INT32, {1}});
  m.SetInput<float>({1.0, 3.0, 5.0});
  m.SetPositions<int32_t>({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({3.0})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1}));
}

TEST(GatherOpTest, Test2DIndexWith2DResult) {
  GatherOpModel m({TensorType_FLOAT32, {3}}, {TensorType_INT32, {1, 2}});
  m.SetInput<float>({1.0, 2.0, 3.0});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({2.0, 1.0})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(FloatGatherOpTest, Duplicate) {
  GatherOpModel m({TensorType_FLOAT32, {1, 2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<float>({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions<int32_t>({0, 0});
  m.Invoke();
  EXPECT_THAT(
      m.GetOutput<float>(),
      ElementsAreArray(ArrayFloatNear({-2, 0.2, 0.7, 0.8, -2, 0.2, 0.7, 0.8})));
}

TEST(FloatGatherOpTest, Slice) {
  GatherOpModel m({TensorType_FLOAT32, {4, 1}}, {TensorType_INT32, {2}});
  m.SetInput<float>({-2.0, 0.2, 0.7, 0.8});
  m.SetPositions<int32_t>({1, 3});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({0.2, 0.8})));
}

TEST(FloatGatherOpTest, Axis1) {
  const int axis = 1;
  GatherOpModel m({TensorType_FLOAT32, {1, 2, 3}}, {TensorType_INT32, {2}},
                  axis);
  m.SetInput<float>({1, 2, 3, 4, 5, 6});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({4, 5, 6, 1, 2, 3})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 3}));
}

TEST(FloatGatherOpTest, Axis10DIndex) {
  const int axis = 1;
  GatherOpModel m({TensorType_FLOAT32, {1, 3, 2}}, {TensorType_INT32, {}},
                  axis);
  m.SetInput<float>({1, 2, 3, 4, 5, 6});
  m.SetPositions<int32_t>({1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({3, 4})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(FloatGatherOpTest, Axis1Slice) {
  const int axis = 1;
  GatherOpModel m({TensorType_FLOAT32, {1, 4, 2}}, {TensorType_INT32, {2}},
                  axis);
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8});
  m.SetPositions<int32_t>({3, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({7, 8, 3, 4})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2}));
}

TEST(FloatGatherOpTest, LastAxis) {
  const int axis = -1;
  GatherOpModel m({TensorType_FLOAT32, {1, 2, 3}}, {TensorType_INT32, {2}},
                  axis);
  m.SetInput<float>({1, 2, 3, 4, 5, 6});
  m.SetPositions<int32_t>({2, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(ArrayFloatNear({3, 1, 6, 4})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2}));
}

TEST(FloatGatherOpTest, LastAxis0DIndex) {
  const int axis = -1;
  GatherOpModel m({TensorType_FLOAT32, {1, 2, 3}}, {TensorType_INT32, {}},
                  axis);
  m.SetInput<float>({1, 2, 3, 4, 5, 6});
  m.SetPositions<int32_t>({2});
  m.Invoke();
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray(ArrayFloatNear({3, 6})));
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2}));
}

TEST(TypesGatherOpTest, Float32Int32) {
  GatherOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<float>({13.3, -13.4, -1.4, 1.5});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.4, 1.5, 13.3, -13.4}));
}

TEST(TypesGatherOpTest, Float32Int64) {
  GatherOpModel m({TensorType_FLOAT32, {2, 2}}, {TensorType_INT64, {2}});
  m.SetInput<float>({13.3, -13.4, -1.4, 1.5});
  m.SetPositions<int64_t>({1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({-1.4, 1.5, 13.3, -13.4}));
}

TEST(TypesGatherOpTest, Int32Int32) {
  GatherOpModel m({TensorType_INT32, {2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<int32_t>({-1330, 1340, 140, -150});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({140, -150, -1330, 1340}));
}

TEST(TypesGatherOpTest, Int32Int64) {
  GatherOpModel m({TensorType_INT32, {2, 2}}, {TensorType_INT64, {2}});
  m.SetInput<int32_t>({-1330, 1340, 140, -150});
  m.SetPositions<int64_t>({1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int32_t>(),
              ElementsAreArray({140, -150, -1330, 1340}));
}

TEST(TypesGatherOpTest, Uint8Int32) {
  GatherOpModel m({TensorType_UINT8, {2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<uint8_t>({133, 134, 14, 15});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({14, 15, 133, 134}));
}

TEST(TypesGatherOpTest, Uint8Int64) {
  GatherOpModel m({TensorType_UINT8, {2, 2}}, {TensorType_INT64, {2}});
  m.SetInput<uint8_t>({133, 134, 14, 15});
  m.SetPositions<int64_t>({1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<uint8_t>(), ElementsAreArray({14, 15, 133, 134}));
}

TEST(TypesGatherOpTest, Int8Int32) {
  GatherOpModel m({TensorType_INT8, {2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<int8_t>({-13, -120, 14, 15});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({14, 15, -13, -120}));
}

TEST(TypesGatherOpTest, Int8Int64) {
  GatherOpModel m({TensorType_INT8, {2, 2}}, {TensorType_INT64, {2}});
  m.SetInput<int8_t>({-13, -120, 14, 15});
  m.SetPositions<int64_t>({1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int8_t>(), ElementsAreArray({14, 15, -13, -120}));
}

TEST(TypesGatherOpTest, Int64Int32) {
  GatherOpModel m({TensorType_INT64, {2, 2}}, {TensorType_INT32, {2}});
  m.SetInput<int64_t>({-(1LL << 34), 134LL, 14LL, 15LL});
  m.SetPositions<int32_t>({1, 0});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({14LL, 15LL, -(1LL << 34), 134LL}));
}

TEST(TypesGatherOpTest, Int64Int64) {
  GatherOpModel m({TensorType_INT64, {2, 2}}, {TensorType_INT64, {2}});
  m.SetInput<int64_t>({-(1LL << 34), 134LL, 14LL, 15LL});
  m.SetPositions<int64_t>({1LL, 0LL});
  m.Invoke();

  EXPECT_THAT(m.GetOutput<int64_t>(),
              ElementsAreArray({14LL, 15LL, -(1LL << 34), 134LL}));
}

TEST(GatherOpTest, SimpleString) {
  GatherOpModel m({TensorType_STRING, {3}}, {TensorType_INT32, {2}});
  m.SetStringInput({"A", "B", "C"});
  m.SetPositions<int32_t>({0, 2});
  m.Invoke();
  ASSERT_THAT(m.GetOutputShape(), ElementsAreArray({2}));
  EXPECT_THAT(m.GetStringOutput(), ElementsAreArray({"A", "C"}));
}
}  // namespace
}  // namespace tflite
