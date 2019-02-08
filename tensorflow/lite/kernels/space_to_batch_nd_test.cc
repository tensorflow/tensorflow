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
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;
using ::testing::Matcher;

class SpaceToBatchNDOpModel : public SingleOpModel {
 public:
  void SetInput(std::initializer_list<float> data) {
    PopulateTensor<float>(input_, data);
  }

  void SetQuantizedInput(std::initializer_list<float> data) {
    QuantizeAndPopulate<uint8_t>(input_, data);
  }

  void SetBlockShape(std::initializer_list<int> data) {
    PopulateTensor<int>(block_shape_, data);
  }

  void SetPaddings(std::initializer_list<int> data) {
    PopulateTensor<int>(paddings_, data);
  }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<float> GetDequantizedOutput() {
    return Dequantize<uint8_t>(ExtractVector<uint8_t>(output_),
                               GetScale(output_), GetZeroPoint(output_));
  }

 protected:
  int input_;
  int block_shape_;
  int paddings_;
  int output_;
};

// Tests case where block_shape and paddings are const tensors.
//
// Example usage is as follows:
//    SpaceToBatchNDOpConstModel m(input_shape, block_shape, paddings);
//    m.SetInput(input_data);
//    m.Invoke();
class SpaceToBatchNDOpConstModel : public SpaceToBatchNDOpModel {
 public:
  SpaceToBatchNDOpConstModel(const TensorData& input,
                             std::initializer_list<int> block_shape,
                             std::initializer_list<int> paddings,
                             const TensorData& output) {
    input_ = AddInput(input);
    block_shape_ = AddConstInput(TensorType_INT32, block_shape, {2});
    paddings_ = AddConstInput(TensorType_INT32, paddings, {2, 2});
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_SPACE_TO_BATCH_ND,
                 BuiltinOptions_SpaceToBatchNDOptions,
                 CreateSpaceToBatchNDOptions(builder_).Union());
    BuildInterpreter({input.shape});
  }
};

// Tests case where block_shape and paddings are non-const tensors.
//
// Example usage is as follows:
//    SpaceToBatchNDOpDynamicModel m(input_shape);
//    m.SetInput(input_data);
//    m.SetBlockShape(block_shape);
//    m.SetPaddings(paddings);
//    m.Invoke();
class SpaceToBatchNDOpDynamicModel : public SpaceToBatchNDOpModel {
 public:
  SpaceToBatchNDOpDynamicModel(const TensorData& input,
                               const TensorData& output) {
    input_ = AddInput(input);
    block_shape_ = AddInput(TensorType_INT32);
    paddings_ = AddInput(TensorType_INT32);
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_SPACE_TO_BATCH_ND,
                 BuiltinOptions_SpaceToBatchNDOptions,
                 CreateSpaceToBatchNDOptions(builder_).Union());
    BuildInterpreter({input.shape, {2}, {2, 2}});
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST(SpaceToBatchNDOpTest, InvalidShapeTest) {
  EXPECT_DEATH(
      SpaceToBatchNDOpConstModel({TensorType_FLOAT32, {1, 3, 3, 1}}, {2, 2},
                                 {0, 0, 0, 0}, {TensorType_FLOAT32}),
      "Cannot allocate tensors");
}
#endif

TEST(SpaceToBatchNDOpTest, SimpleConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 4, 4, 1}}, {2, 2},
                               {0, 0, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, SimpleDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 4, 4, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetPaddings({0, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({4, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, MultipleInputBatchesConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {2, 2, 4, 1}}, {2, 2},
                               {0, 0, 0, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, MultipleInputBatchesDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {2, 2, 4, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetPaddings({0, 0, 0, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({8, 1, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({1, 3, 9, 11, 2, 4, 10, 12, 5, 7,
                                               13, 15, 6, 8, 14, 16}));
}

TEST(SpaceToBatchNDOpTest, SimplePaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 5, 2, 1}}, {3, 2},
                               {1, 0, 2, 0}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
                                 0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10,
                             }));
}

TEST(SpaceToBatchNDOpTest, SimplePaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 5, 2, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 0, 2, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 5, 0, 0, 0, 6, 0, 1, 0, 7,
                                 0, 2, 0, 8, 0, 3, 0, 9, 0, 4, 0, 10,
                             }));
}

TEST(SpaceToBatchNDOpTest, ComplexPaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_FLOAT32, {1, 4, 2, 1}}, {3, 2},
                               {1, 1, 2, 4}, {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
                                 0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
                                 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
                             }));
}

TEST(SpaceToBatchNDOpTest, ComplexPaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_FLOAT32, {1, 4, 2, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput({1, 2, 3, 4, 5, 6, 7, 8});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 1, 2, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetOutput(), ElementsAreArray({
                                 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
                                 0, 1, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0,
                                 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
                             }));
}

class QuantizedSpaceToBatchNDOpTest : public ::testing::Test {
 protected:
  std::vector<Matcher<float>> DequantizedArrayNear(
      const std::vector<float>& values, const float min, const float max) {
    const float quantization_tolerance = (max - min) / 255.0;
    return ArrayFloatNear(values, quantization_tolerance);
  }
};

#ifdef GTEST_HAS_DEATH_TEST
TEST_F(QuantizedSpaceToBatchNDOpTest, ZeroNotInQuantizationRange) {
  // The test_util and actual quantization code currently ensure that the range
  // must include zero, but if that ever changes, this test will catch it.
  EXPECT_DEATH(SpaceToBatchNDOpConstModel m(
                   {TensorType_UINT8, {1, 2, 2, 1}, 1.0, 2.0}, {4, 2},
                   {0, 0, 1, 1, 1, 1, 0, 0}, {TensorType_UINT8, {}, 1.0, 2.0}),
               ".*Check failed: f_min <= 0.*");
}
#endif

TEST_F(QuantizedSpaceToBatchNDOpTest, SimplePaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_UINT8, {1, 5, 2, 1}, -1.0, 1.0},
                               {3, 2}, {1, 0, 2, 0},
                               {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -0.1, 0, -0.7,
                   0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 0.1},
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, SimplePaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_UINT8, {1, 5, 2, 1}, -1.0, 1.0},
                                 {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 0.1});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 0, 2, 0});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 2, 1}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(DequantizedArrayNear(
                  {0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -0.1, 0, -0.7,
                   0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 0.1},
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, ComplexPaddingConstTest) {
  SpaceToBatchNDOpConstModel m({TensorType_UINT8, {1, 4, 2, 1}, -1.0, 1.0},
                               {3, 2}, {1, 1, 2, 4},
                               {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(DequantizedArrayNear(
                  {
                      0, 0,    0, 0, 0, -0.5, 0, 0, 0, 0,   0, 0, 0, 0.6, 0, 0,
                      0, -0.1, 0, 0, 0, -0.7, 0, 0, 0, 0.2, 0, 0, 0, 0.8, 0, 0,
                      0, -0.3, 0, 0, 0, 0,    0, 0, 0, 0.4, 0, 0, 0, 0,   0, 0,
                  },
                  -1.0, 1.0)));
}

TEST_F(QuantizedSpaceToBatchNDOpTest, ComplexPaddingDynamicTest) {
  SpaceToBatchNDOpDynamicModel m({TensorType_UINT8, {1, 4, 2, 1}, -1.0, 1.0},
                                 {TensorType_UINT8, {}, -1.0, 1.0});
  m.SetQuantizedInput({-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8});
  m.SetBlockShape({3, 2});
  m.SetPaddings({1, 1, 2, 4});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({6, 2, 4, 1}));
  EXPECT_THAT(m.GetDequantizedOutput(),
              ElementsAreArray(DequantizedArrayNear(
                  {
                      0, 0,    0, 0, 0, -0.5, 0, 0, 0, 0,   0, 0, 0, 0.6, 0, 0,
                      0, -0.1, 0, 0, 0, -0.7, 0, 0, 0, 0.2, 0, 0, 0, 0.8, 0, 0,
                      0, -0.3, 0, 0, 0, 0,    0, 0, 0, 0.4, 0, 0, 0, 0,   0, 0,
                  },
                  -1.0, 1.0)));
}

}  // namespace
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
