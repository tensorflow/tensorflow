/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/elementwise.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/test_util.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace gl {
namespace {

TensorRef<BHWC> GetTensorRef(int ref, const BHWC& shape) {
  TensorRef<BHWC> tensor_ref;
  tensor_ref.type = DataType::FLOAT32;
  tensor_ref.ref = ref;
  tensor_ref.shape = shape;
  return tensor_ref;
}

TEST(ElementwiseTest, Abs) {
  OperationType op_type = OperationType::ABS;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 6.2, 2.0, 4.0}));
}

TEST(ElementwiseTest, Cos) {
  OperationType op_type = OperationType::COS;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 3.1415926, -3.1415926, 1}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, -1.0, -1.0, 0.540302}));
}

TEST(ElementwiseTest, Div) {
  OperationType op_type = OperationType::DIV;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {1.0, 2.0, -0.5, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, -3.1, -4.0, 1.0}));
}

TEST(ElementwiseTest, HardSwish) {
  OperationType op_type = OperationType::HARD_SWISH;
  const BHWC shape(1, 1, 1, 7);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(
      model.PopulateTensor(0, {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6f),
                        {0.0f, 0.0f, -0.375f, 0.0f, 1.125f, 3.f, 4.5f}));
}

TEST(ElementwiseTest, Log) {
  OperationType op_type = OperationType::LOG;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 3.1415926, 1.0, 1.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.14473, 0.0, 0.0}));
}

TEST(ElementwiseTest, Pow) {
  OperationType op_type = OperationType::POW;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.0, 8.0, 256.0}));
}

TEST(ElementwiseTest, Rsqrt) {
  OperationType op_type = OperationType::RSQRT;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 4.0, 9.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 0.707106, 0.5, 0.333333}));
}

TEST(ElementwiseTest, Sigmoid) {
  OperationType op_type = OperationType::SIGMOID;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.0, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.5, 0.002473, 0.880797, 0.982014}));
}

TEST(ElementwiseTest, Sin) {
  OperationType op_type = OperationType::SIN;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 3.1415926, -3.1415926, 1.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 0.0, 0.0, 0.841471}));
}

TEST(ElementwiseTest, Sqrt) {
  OperationType op_type = OperationType::SQRT;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.0, 1.414213, 2.0}));
}

TEST(ElementwiseTest, Square) {
  OperationType op_type = OperationType::SQUARE;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 0.5, -3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 4.0, 0.25, 9.0}));
}

TEST(ElementwiseTest, SquaredDiff) {
  OperationType op_type = OperationType::SQUARED_DIFF;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 2.0, 2.0, 4.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {1.0, 1.0, 5.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 1.0, 9.0, 0.0}));
}

TEST(ElementwiseTest, Sub) {
  OperationType op_type = OperationType::SUB;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {1.0, 2.0, 3.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.0, -8.2, -1.0, 0.0}));
}

TEST(ElementwiseTest, Tanh) {
  OperationType op_type = OperationType::TANH;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.0, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, -0.999987, 0.964027, 0.999329}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
