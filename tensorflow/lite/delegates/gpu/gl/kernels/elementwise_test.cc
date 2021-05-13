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

using ::testing::FloatEq;
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

TEST(ElementwiseOneArgumentTest, Abs) {
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

TEST(ElementwiseOneArgumentTest, Cos) {
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

TEST(ElementwiseOneArgumentTest, Copy) {
  OperationType op_type = OperationType::COPY;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatEq(), {0.0, -6.2, 2.0, 4.0}));
}

TEST(ElementwiseOneArgumentTest, Elu) {
  OperationType op_type = OperationType::ELU;
  const BHWC shape(1, 1, 1, 7);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(
      0, {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0f, 1.0f, std::exp(-1.0f) - 1.0f,
                                          100.0f, std::exp(-100.0f) - 1.0f,
                                          0.01f, std::exp(-0.01f) - 1.0f}));
}

TEST(ElementwiseOneArgumentTest, Exp) {
  OperationType op_type = OperationType::EXP;
  const BHWC shape(1, 1, 1, 7);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(
      0, {0.0f, 1.0f, -1.0f, 100.0f, -100.0f, 0.01f, -0.01f}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {std::exp(0.0f), std::exp(1.0f), std::exp(-1.0f),
                         std::exp(100.0f), std::exp(-100.0f), std::exp(0.01f),
                         std::exp(-0.01f)}));
}

TEST(ElementwiseOneArgumentTest, Floor) {
  OperationType op_type = OperationType::FLOOR;
  const BHWC shape(1, 1, 1, 7);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(
      model.PopulateTensor(0, {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {-5.0f, -3.0f, -2.0f, 0.0f, 1.0f, 3.0f, 4.0f}));
}

TEST(ElementwiseOneArgumentTest, HardSwish) {
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

TEST(ElementwiseOneArgumentTest, Log) {
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

TEST(ElementwiseOneArgumentTest, Neg) {
  OperationType op_type = OperationType::NEG;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model({/*type=*/ToString(op_type), /*attributes=*/{}},
                      /*inputs=*/{GetTensorRef(0, shape)},
                      /*outputs=*/{GetTensorRef(1, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, -3.1415926, 0.0, 1.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.0, 3.1415926, 0.0, -1.0}));
}

TEST(ElementwiseOneArgumentTest, Rsqrt) {
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

TEST(ElementwiseOneArgumentTest, Sigmoid) {
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

TEST(ElementwiseOneArgumentTest, Sin) {
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

TEST(ElementwiseOneArgumentTest, Sqrt) {
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

TEST(ElementwiseOneArgumentTest, Square) {
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

TEST(ElementwiseOneArgumentTest, Tanh) {
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

TEST(ElementwiseTwoArgumentsTest, DivElementwise) {
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

TEST(ElementwiseTwoArgumentsTest, DivBroadcast) {
  OperationType op_type = OperationType::DIV;
  const BHWC shape0(1, 2, 1, 2);
  const BHWC shape1(1, 1, 1, 2);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape0), GetTensorRef(1, shape1)},
      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {0.5, 0.2}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 5.0, 4.0, 15.0}));
}

TEST(ElementwiseTwoArgumentsTest, DivScalar) {
  OperationType op_type = OperationType::DIV;
  const BHWC shape0(1, 2, 1, 2);
  ElementwiseAttributes attr;
  attr.param = static_cast<float>(0.5);
  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 2.0, 4.0, 6.0}));
}

TEST(ElementwiseTwoArgumentsTest, DivConstVector) {
  OperationType op_type = OperationType::DIV;
  const BHWC shape0(1, 2, 1, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> param;
  param.shape = Linear(2);
  param.id = 1;
  param.data = {0.4, 0.5};
  attr.param = std::move(param);

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 2.0, 5.0, 6.0}));
}

TEST(ElementwiseTwoArgumentsTest, FloorDiv) {
  OperationType op_type = OperationType::FLOOR_DIV;
  const BHWC shape0(1, 1, 1, 7);

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(
      model.PopulateTensor(0, {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {std::floor(-4.5f / scalar), std::floor(-3.0f / scalar),
                         std::floor(-1.5f / scalar), std::floor(0.0f / scalar),
                         std::floor(1.5f / scalar), std::floor(3.0f / scalar),
                         std::floor(4.5f / scalar)}));
}

TEST(ElementwiseTwoArgumentsTest, FloorMod) {
  OperationType op_type = OperationType::FLOOR_MOD;
  const BHWC shape0(1, 1, 1, 7);

  float scalar = 2.7f;
  ElementwiseAttributes attr;
  attr.param = scalar;

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(
      model.PopulateTensor(0, {-4.5f, -3.0f, -1.5f, 0.0f, 1.5f, 3.0f, 4.5f}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {-4.5f - std::floor(-4.5f / scalar) * scalar,
                                  -3.0f - std::floor(-3.0f / scalar) * scalar,
                                  -1.5f - std::floor(-1.5f / scalar) * scalar,
                                  0.0f - std::floor(0.0f / scalar) * scalar,
                                  1.5f - std::floor(1.5f / scalar) * scalar,
                                  3.0f - std::floor(3.0f / scalar) * scalar,
                                  4.5f - std::floor(4.5f / scalar) * scalar}));
}

TEST(ElementwiseTwoArgumentsTest, MaximumElementwise) {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {1.0, 2.0, 3.0, -2.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 2.0, 3.0, -2.0}));
}

TEST(ElementwiseTwoArgumentsTest, MaximumBroadcast) {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape0(1, 2, 1, 2);
  const BHWC shape1(1, 1, 1, 2);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape0), GetTensorRef(1, shape1)},
      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {0.5, 0.2}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.5, 1.0, 2.0, 3.0}));
}

TEST(ElementwiseTwoArgumentsTest, MaximumScalar) {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape(1, 2, 2, 1);
  ElementwiseAttributes attr;
  attr.param = -1.0f;
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/std::move(attr)},
      /*inputs=*/{GetTensorRef(0, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, -1.0, 2.0, -1.0}));
}

TEST(ElementwiseTwoArgumentsTest, MaximumConstVector) {
  OperationType op_type = OperationType::MAXIMUM;
  const BHWC shape0(1, 2, 1, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> param;
  param.shape = Linear(2);
  param.id = 1;
  param.data = {0.4, 0.5};
  attr.param = std::move(param);

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.4, 1.0, 2.0, 3.0}));
}

TEST(ElementwiseTwoArgumentsTest, MinimumElementwise) {
  OperationType op_type = OperationType::MINIMUM;
  const BHWC shape(1, 2, 2, 1);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape), GetTensorRef(1, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {1.0, 2.0, 3.0, -2.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, -6.2, 2.0, -3.0}));
}

TEST(ElementwiseTwoArgumentsTest, MinimumBroadcast) {
  OperationType op_type = OperationType::MINIMUM;
  const BHWC shape0(1, 2, 1, 2);
  const BHWC shape1(1, 1, 1, 2);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape0), GetTensorRef(1, shape1)},
      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {0.5, 0.2}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 0.2, 0.5, 0.2}));
}

TEST(ElementwiseTwoArgumentsTest, MinimumScalar) {
  OperationType op_type = OperationType::MINIMUM;
  const BHWC shape(1, 2, 2, 1);
  ElementwiseAttributes attr;
  attr.param = -1.0f;
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/std::move(attr)},
      /*inputs=*/{GetTensorRef(0, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, -3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-1.0, -6.2, -1.0, -3.0}));
}

TEST(ElementwiseTwoArgumentsTest, MinimumConstVector) {
  OperationType op_type = OperationType::MINIMUM;
  const BHWC shape0(1, 2, 1, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> param;
  param.shape = Linear(2);
  param.id = 1;
  param.data = {0.5, 0.2};
  attr.param = std::move(param);

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 0.2, 0.5, 0.2}));
}

TEST(ElementwiseTwoArgumentsTest, PowElementwise) {
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

TEST(ElementwiseTwoArgumentsTest, PowBroadcast) {
  OperationType op_type = OperationType::POW;
  const BHWC shape0(1, 2, 1, 2);
  const BHWC shape1(1, 1, 1, 2);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape0), GetTensorRef(1, shape1)},
      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {2.0, 0.5}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.0, 4.0, 2.0}));
}

TEST(ElementwiseTwoArgumentsTest, PowScalar) {
  OperationType op_type = OperationType::POW;
  const BHWC shape(1, 2, 2, 1);
  ElementwiseAttributes attr;
  attr.param = 2.0f;
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/std::move(attr)},
      /*inputs=*/{GetTensorRef(0, shape)},
      /*outputs=*/{GetTensorRef(2, shape)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.0, 4.0, 16.0}));
}

TEST(ElementwiseTwoArgumentsTest, PowConstVector) {
  OperationType op_type = OperationType::POW;
  const BHWC shape0(1, 2, 1, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> param;
  param.shape = Linear(2);
  param.id = 1;
  param.data = {2.0, 0.5};
  attr.param = std::move(param);

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.0, 4.0, 2.0}));
}

TEST(ElementwiseTwoArgumentsTest, SquaredDiffElementwise) {
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

TEST(ElementwiseTwoArgumentsTest, SquaredDiffBroadcast) {
  OperationType op_type = OperationType::SQUARED_DIFF;
  const BHWC shape0(1, 2, 1, 2);
  const BHWC shape1(1, 1, 1, 2);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape0), GetTensorRef(1, shape1)},
      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {-1.0, 5.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 16.0, 9.0, 4.0}));
}

TEST(ElementwiseTwoArgumentsTest, SquaredDiffScalar) {
  OperationType op_type = OperationType::SQUARED_DIFF;
  const BHWC shape0(1, 2, 1, 2);
  ElementwiseAttributes attr;
  attr.param = static_cast<float>(5.0);
  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {25.0, 16.0, 9.0, 4.0}));
}

TEST(ElementwiseTwoArgumentsTest, SquaredDiffConstVector) {
  OperationType op_type = OperationType::SQUARED_DIFF;
  const BHWC shape0(1, 2, 1, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> param;
  param.shape = Linear(2);
  param.id = 1;
  param.data = {-1.0, 5.0};
  attr.param = std::move(param);

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 16.0, 9.0, 4.0}));
}

TEST(ElementwiseTwoArgumentsTest, SubElementwise) {
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

TEST(ElementwiseTwoArgumentsTest, SubBroadcast) {
  OperationType op_type = OperationType::SUB;
  const BHWC shape0(1, 2, 1, 2);
  const BHWC shape1(1, 1, 1, 2);
  SingleOpModel model(
      {/*type=*/ToString(op_type), /*attributes=*/{}},
      /*inputs=*/{GetTensorRef(0, shape0), GetTensorRef(1, shape1)},
      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_TRUE(model.PopulateTensor(1, {0.3, 0.2}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-0.3, 0.8, 1.7, 2.8}));
}

TEST(ElementwiseTwoArgumentsTest, SubScalar) {
  OperationType op_type = OperationType::SUB;
  const BHWC shape0(1, 2, 1, 2);
  ElementwiseAttributes attr;
  attr.param = static_cast<float>(0.5);
  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-0.5, 0.5, 1.5, 2.5}));
}

TEST(ElementwiseTwoArgumentsTest, SubConstVector) {
  OperationType op_type = OperationType::SUB;
  const BHWC shape0(1, 2, 1, 2);

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> param;
  param.shape = Linear(2);
  param.id = 1;
  param.data = {0.3, 0.2};
  attr.param = std::move(param);

  SingleOpModel model({/*type=*/ToString(op_type), attr},
                      /*inputs=*/{GetTensorRef(0, shape0)},
                      /*outputs=*/{GetTensorRef(2, shape0)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {-0.3, 0.8, 1.7, 2.8}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
