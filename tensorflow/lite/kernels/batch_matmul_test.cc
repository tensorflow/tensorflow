/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class BatchMatMulOpModel : public SingleOpModel {
 public:
  BatchMatMulOpModel(const TensorData& lhs, const TensorData& rhs) {
    lhs_id_ = AddInput(lhs);
    rhs_id_ = AddInput(rhs);
    output_id_ = AddOutput(lhs.type);
    SetBuiltinOp(BuiltinOperator_BATCH_MATMUL, BuiltinOptions_NONE, 0);
    BuildInterpreter({GetShape(lhs_id_), GetShape(rhs_id_)});
  }

  int lhs() const { return lhs_id_; }
  int rhs() const { return rhs_id_; }
  std::vector<T> GetOutput() { return ExtractVector<T>(output_id_); }
  std::vector<int32_t> GetOutputShape() { return GetTensorShape(output_id_); }

 protected:
  int lhs_id_;
  int rhs_id_;
  int output_id_;
};

TEST(BatchMatMulOpModelTest, Float32Test_Simple) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 2, 3}},
                                  {TensorType_FLOAT32, {1, 3, 4}});
  model.PopulateTensor<float>(model.lhs(), {1, 2, 3, 4, 5, 6});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({50.0f, 122.0f, 68.0f, 167.0f, 86.0f, 212.0f,
                                104.0f, 257.0f}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_BatchSizeTwo) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 2, 3}},
                                  {TensorType_FLOAT32, {2, 3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});
  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({50.0f, 122.0f, 68.0f, 167.0f, 86.0f, 212.0f,
                                104.0f, 257.0f, 482.0f, 662.0f, 554.0f, 761.0f,
                                626.0f, 860.0f, 698.0f, 959.0f}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 2, 3}},
                                  {TensorType_FLOAT32, {3, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

  model.Invoke();
  EXPECT_THAT(model.GetOutput(),
              ElementsAreArray({50.0f, 122.0f, 68.0f, 167.0f, 86.0f, 212.0f,
                                104.0f, 257.0f, 194.0f, 266.0f, 266.0f, 365.0f,
                                338.0f, 464.0f, 410.0f, 563.0f}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 2, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_Broadcast2) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {2, 1, 3, 2}},
                                  {TensorType_FLOAT32, {3, 2, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(
          {23.0f,  53.0f,  83.0f,  29.0f,  67.0f,  105.0f, 35.0f,  81.0f,
           127.0f, 41.0f,  95.0f,  149.0f, 47.0f,  109.0f, 171.0f, 53.0f,
           123.0f, 193.0f, 59.0f,  137.0f, 215.0f, 65.0f,  151.0f, 237.0f,
           71.0f,  165.0f, 259.0f, 77.0f,  179.0f, 281.0f, 83.0f,  193.0f,
           303.0f, 89.0f,  207.0f, 325.0f, 113.0f, 143.0f, 173.0f, 143.0f,
           181.0f, 219.0f, 173.0f, 219.0f, 265.0f, 203.0f, 257.0f, 311.0f,
           233.0f, 295.0f, 357.0f, 263.0f, 333.0f, 403.0f, 293.0f, 371.0f,
           449.0f, 323.0f, 409.0f, 495.0f, 353.0f, 447.0f, 541.0f, 383.0f,
           485.0f, 587.0f, 413.0f, 523.0f, 633.0f, 443.0f, 561.0f, 679.0f}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({2, 3, 3, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_BroadcastFiveD) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {1, 2, 1, 3, 2}},
                                  {TensorType_FLOAT32, {3, 2, 4}});
  model.PopulateTensor<float>(model.lhs(),
                              {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  model.PopulateTensor<float>(model.rhs(),
                              {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
                               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray(
          {23.0f,  53.0f,  83.0f,  29.0f,  67.0f,  105.0f, 35.0f,  81.0f,
           127.0f, 41.0f,  95.0f,  149.0f, 47.0f,  109.0f, 171.0f, 53.0f,
           123.0f, 193.0f, 59.0f,  137.0f, 215.0f, 65.0f,  151.0f, 237.0f,
           71.0f,  165.0f, 259.0f, 77.0f,  179.0f, 281.0f, 83.0f,  193.0f,
           303.0f, 89.0f,  207.0f, 325.0f, 113.0f, 143.0f, 173.0f, 143.0f,
           181.0f, 219.0f, 173.0f, 219.0f, 265.0f, 203.0f, 257.0f, 311.0f,
           233.0f, 295.0f, 357.0f, 263.0f, 333.0f, 403.0f, 293.0f, 371.0f,
           449.0f, 323.0f, 409.0f, 495.0f, 353.0f, 447.0f, 541.0f, 383.0f,
           485.0f, 587.0f, 413.0f, 523.0f, 633.0f, 443.0f, 561.0f, 679.0f}));

  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({1, 2, 3, 3, 4}));
}

TEST(BatchMatMulOpModelTest, Float32Test_BroadcastFromRHS) {
  BatchMatMulOpModel<float> model({TensorType_FLOAT32, {4, 5}},
                                  {TensorType_FLOAT32, {3, 1, 5, 2}});
  model.PopulateTensor<float>(
      model.lhs(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  model.PopulateTensor<float>(
      model.rhs(),
      {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
       22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});

  model.Invoke();
  EXPECT_THAT(
      model.GetOutput(),
      ElementsAreArray({145.0f,  370.0f,  595.0f,  820.0f,  220.0f,  570.0f,
                        920.0f,  1270.0f, 295.0f,  770.0f,  1245.0f, 1720.0f,
                        370.0f,  970.0f,  1570.0f, 2170.0f, 445.0f,  1170.0f,
                        1895.0f, 2620.0f, 520.0f,  1370.0f, 2220.0f, 3070.0f}));
  EXPECT_THAT(model.GetOutputShape(), ElementsAreArray({3, 1, 4, 2}));
}

}  // namespace
}  // namespace tflite
