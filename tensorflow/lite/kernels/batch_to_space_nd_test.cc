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

#include <stdint.h>

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

class BatchToSpaceNDOpModel : public SingleOpModel {
 public:
  template <typename T>
  void SetInput(std::initializer_list<T> data) {
    PopulateTensor<T>(input_, data);
  }

  void SetBlockShape(std::initializer_list<int> data) {
    PopulateTensor<int>(block_shape_, data);
  }

  void SetCrops(std::initializer_list<int> data) {
    PopulateTensor<int>(crops_, data);
  }

  int input_tensor_id() { return input_; }

  template <typename T>
  std::vector<T> GetOutput() {
    return ExtractVector<T>(output_);
  }

  template <typename T>
  std::vector<float> GetDequantizedOutput() {
    return Dequantize<T>(ExtractVector<T>(output_), GetScale(output_),
                         GetZeroPoint(output_));
  }

  int32_t GetOutputSize() { return GetTensorSize(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 protected:
  int input_;
  int block_shape_;
  int crops_;
  int output_;
};

// Tests case where block_shape and crops are const tensors.
//
// Example usage is as follows:
//    BatchToSpaceNDOpConstModel m(input_shape, block_shape, crops);
//    m.SetInput(input_data);
//    m.Invoke();
class BatchToSpaceNDOpConstModel : public BatchToSpaceNDOpModel {
 public:
  BatchToSpaceNDOpConstModel(const TensorData& input,
                             std::initializer_list<int> block_shape,
                             std::initializer_list<int> crops,
                             const TensorData& output) {
    int spatial_dims = static_cast<int>(block_shape.size());
    input_ = AddInput(input);
    block_shape_ = AddConstInput(TensorType_INT32, block_shape, {spatial_dims});
    crops_ = AddConstInput(TensorType_INT32, crops, {spatial_dims, 2});
    output_ = AddOutput(output);

    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({GetShape(input_)});
  }
};

// Tests case where block_shape and crops are non-const tensors.
//
// Example usage is as follows:
//    BatchToSpaceNDOpDynamicModel m(input_shape);
//    m.SetInput(input_data);
//    m.SetBlockShape(block_shape);
//    m.SetPaddings(crops);
//    m.Invoke();
class BatchToSpaceNDOpDynamicModel : public BatchToSpaceNDOpModel {
 public:
  BatchToSpaceNDOpDynamicModel(const TensorData& input,
                               const TensorData& output) {
    input_ = AddInput(input);
    block_shape_ = AddInput(TensorType_INT32);
    crops_ = AddInput(TensorType_INT32);
    output_ = AddOutput(output);

    int spatial_dims = static_cast<int>(GetShape(input_).size()) - 2;
    SetBuiltinOp(BuiltinOperator_BATCH_TO_SPACE_ND,
                 BuiltinOptions_BatchToSpaceNDOptions,
                 CreateBatchToSpaceNDOptions(builder_).Union());
    BuildInterpreter({GetShape(input_), {spatial_dims}, {spatial_dims, 2}});
  }
};

template <typename integer_type>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      (max - min) / (std::numeric_limits<integer_type>::max() -
                     std::numeric_limits<integer_type>::min());
  return kQuantizedStep;
}

TEST(BatchToSpaceNDOpTest, SimpleConstTest) {
  BatchToSpaceNDOpConstModel m({TensorType_FLOAT32, {4, 2, 2, 1}}, {2, 2},
                               {0, 0, 0, 0}, {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, SimpleConstTestInt8) {
  BatchToSpaceNDOpConstModel m({TensorType_INT8, {4, 2, 2, 1}}, {2, 2},
                               {0, 0, 0, 0}, {TensorType_INT8});
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, BatchOneConstTest) {
  BatchToSpaceNDOpConstModel m({TensorType_FLOAT32, {1, 2, 2, 1}}, {1, 1},
                               {0, 0, 0, 0}, {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 2, 2, 1}));
  EXPECT_THAT(m.GetOutput<float>(), ElementsAreArray({1, 2, 3, 4}));
}

TEST(BatchToSpaceNDOpTest, SimpleConstTestInt8EmptyOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    // NNAPI doesn't currently support non-zero crop values.
    return;
  }

  BatchToSpaceNDOpConstModel m({TensorType_INT8, {4, 2, 2, 1}}, {2, 2},
                               {0, 0, 2, 2}, {TensorType_INT8});
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 0, 1}));
  EXPECT_THAT(m.GetOutputSize(), 0);
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({TensorType_FLOAT32, {4, 2, 2, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTestInt8) {
  BatchToSpaceNDOpDynamicModel m({TensorType_INT8, {4, 2, 2, 1}},
                                 {TensorType_INT8});
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(),
              ElementsAreArray(
                  {1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16}));
}

TEST(BatchToSpaceNDOpTest, InvalidCropsDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({TensorType_FLOAT32, {4, 2, 2, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({0, 0, -1, 0});
  ASSERT_NE(m.Invoke(), kTfLiteOk) << "crops.i. >= 0 was not true.";
}

TEST(BatchToSpaceNDOpTest, SimpleDynamicTestInt8EmptyOutput) {
  if (SingleOpModel::GetForceUseNnapi()) {
    // NNAPI doesn't currently support non-zero crop values.
    return;
  }

  BatchToSpaceNDOpDynamicModel m({TensorType_INT8, {4, 2, 2, 1}},
                                 {TensorType_INT8});
  m.SetInput<int8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2, 2});
  m.SetCrops({2, 2, 0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({1, 0, 4, 1}));
  EXPECT_THAT(m.GetOutput<int8_t>(), ::testing::IsEmpty());
}

#if GTEST_HAS_DEATH_TEST
TEST(BatchToSpaceNDOpTest, InvalidShapeTest) {
  EXPECT_DEATH(
      BatchToSpaceNDOpConstModel({TensorType_FLOAT32, {3, 2, 2, 1}}, {2, 2},
                                 {0, 0, 0, 0}, {TensorType_FLOAT32}),
      "Cannot allocate tensors");
}

TEST(BatchToSpaceNDOpTest, InvalidCropsConstTest) {
  EXPECT_DEATH(
      BatchToSpaceNDOpConstModel({TensorType_FLOAT32, {3, 2, 2, 1}}, {2, 2},
                                 {0, 0, 0, -1}, {TensorType_FLOAT32}),
      "crops.i. >= 0 was not true.");
}
#endif

TEST(BatchToSpaceNDOpTest, Simple3DConstTest) {
  BatchToSpaceNDOpConstModel m({TensorType_FLOAT32, {4, 4, 1}}, {2}, {0, 0},
                               {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 8, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16}));
}

TEST(BatchToSpaceNDOpTest, Simple3DConstTestWithCrops) {
  BatchToSpaceNDOpConstModel m({TensorType_FLOAT32, {4, 4, 1}}, {2}, {1, 1},
                               {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8}));
}

template <typename integer_dtype>
void Simple3DConstTestWithCropsQuant() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-16.0, 16.0);
  BatchToSpaceNDOpConstModel m(
      {GetTensorType<integer_dtype>(), {4, 4, 1}, 16.0 * kMin, 16.0 * kMax},
      {2}, {1, 1},
      {GetTensorType<integer_dtype>(), {}, 16.0 * kMin, 16.0 * kMax});
  m.QuantizeAndPopulate<integer_dtype>(
      m.input_tensor_id(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8},
                                      kQuantizedTolerance)));
}

TEST(BatchToSpaceNDOpTest, Simple3DConstTestWithCropsUINT8) {
  Simple3DConstTestWithCropsQuant<uint8_t>();
}

TEST(BatchToSpaceNDOpTest, Simple3DConstTestWithCropsINT8) {
  Simple3DConstTestWithCropsQuant<int8_t>();
}

TEST(BatchToSpaceNDOpTest, Simple3DConstTestWithCropsINT16) {
  Simple3DConstTestWithCropsQuant<int16_t>();
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTest) {
  BatchToSpaceNDOpDynamicModel m({TensorType_FLOAT32, {4, 4, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetCrops({0, 0});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 8, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray(
                  {1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16}));
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTestWithCrops) {
  BatchToSpaceNDOpDynamicModel m({TensorType_FLOAT32, {4, 4, 1}},
                                 {TensorType_FLOAT32});
  m.SetInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetCrops({1, 1});
  ASSERT_EQ(m.Invoke(), kTfLiteOk);
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 1}));
  EXPECT_THAT(m.GetOutput<float>(),
              ElementsAreArray({9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8}));
}

template <typename integer_dtype>
void Simple3DDynamicTestWithCropsQuant() {
  const float kMin = -1;
  const float kMax =
      std::numeric_limits<integer_dtype>::max() /
      static_cast<float>(std::numeric_limits<integer_dtype>::max() + 1);
  float kQuantizedTolerance = GetTolerance<integer_dtype>(-16.0, 16.0);
  BatchToSpaceNDOpDynamicModel m(
      {GetTensorType<integer_dtype>(), {4, 4, 1}, 16.0 * kMin, 16.0 * kMax},
      {GetTensorType<integer_dtype>(), {}, 16.0 * kMin, 16.0 * kMax});
  m.QuantizeAndPopulate<integer_dtype>(
      m.input_tensor_id(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  m.SetBlockShape({2});
  m.SetCrops({1, 1});
  m.Invoke();
  EXPECT_THAT(m.GetOutputShape(), ElementsAreArray({2, 6, 1}));
  EXPECT_THAT(
      m.GetDequantizedOutput<integer_dtype>(),
      ElementsAreArray(ArrayFloatNear({9, 2, 10, 3, 11, 4, 13, 6, 14, 7, 15, 8},
                                      kQuantizedTolerance)));
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTestWithCropsQuantUINT8) {
  Simple3DDynamicTestWithCropsQuant<uint8_t>();
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTestWithCropsQuantINT8) {
  Simple3DDynamicTestWithCropsQuant<int8_t>();
}

TEST(BatchToSpaceNDOpTest, Simple3DDynamicTestWithCropsQuantINT16) {
  Simple3DDynamicTestWithCropsQuant<int16_t>();
}

}  // namespace
}  // namespace tflite
