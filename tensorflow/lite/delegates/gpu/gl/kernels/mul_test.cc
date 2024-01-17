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

#include "tensorflow/lite/delegates/gpu/gl/kernels/mul.h"

#include <utility>
#include <vector>

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

TEST(MulTest, ConstantTensorMatchingShape) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = input.shape;

  ElementwiseAttributes attr;
  Tensor<HWC, DataType::FLOAT32> tensor_3d;
  tensor_3d.shape.h = input.shape.h;
  tensor_3d.shape.w = input.shape.w;
  tensor_3d.shape.c = input.shape.c;
  tensor_3d.id = 2;
  tensor_3d.data = {-2, 2, -3, 3};
  attr.param = std::move(tensor_3d);

  SingleOpModel model({ToString(OperationType::MUL), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {-2, 4, -9, 12}));
}

TEST(MulTest, ConstantTensorSingleChannel) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = input.shape;

  ElementwiseAttributes attr;
  Tensor<HWC, DataType::FLOAT32> tensor_3d;
  tensor_3d.shape.h = input.shape.h;
  tensor_3d.shape.w = input.shape.w;
  tensor_3d.shape.c = 1;
  tensor_3d.id = 2;
  tensor_3d.data = {-2, 2};
  attr.param = std::move(tensor_3d);

  SingleOpModel model({ToString(OperationType::MUL), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {-2, -4, 6, 8}));
}

TEST(MulTest, DegenerateConstantTensorSingleValue) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = input.shape;

  ElementwiseAttributes attr;
  Tensor<HWC, DataType::FLOAT32> tensor_3d;
  tensor_3d.shape.h = 1;
  tensor_3d.shape.w = 1;
  tensor_3d.shape.c = 1;
  tensor_3d.id = 2;
  tensor_3d.data = {-2};
  attr.param = std::move(tensor_3d);

  SingleOpModel model({ToString(OperationType::MUL), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {-2, -4, -6, -8}));
}

TEST(MulTest, ConstantTensorLinear) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = input.shape;

  ElementwiseAttributes attr;
  Tensor<Linear, DataType::FLOAT32> tensor;
  tensor.shape.v = 2;
  tensor.id = 1;
  tensor.data = {2, 3};
  attr.param = std::move(tensor);

  SingleOpModel model({ToString(OperationType::MUL), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2, 6, 6, 12}));
}

TEST(MulTest, ConstantTensorScalar) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = input.shape;

  ElementwiseAttributes attr;
  attr.param = 2.f;

  SingleOpModel model({ToString(OperationType::MUL), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2, 4, 6, 8}));
}

TEST(MulTest, RuntimeTensorMatchingShapeNonOnes) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 2, 2, 2);

  TensorRef<BHWC> mask;
  mask.type = DataType::FLOAT32;
  mask.ref = 1;
  mask.shape = input.shape;

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = input.shape;

  SingleOpModel model({ToString(OperationType::MUL), {}}, {input, mask},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4, -1, -2, -3, -4}));
  ASSERT_TRUE(model.PopulateTensor(1, {5, 6, 7, 8, 9, 10, 11, 12}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {5, 12, 21, 32, -9, -20, -33, -48}));
}

TEST(MulTest, RuntimeTensorMatchingShapeHeightOne) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> mask;
  mask.type = DataType::FLOAT32;
  mask.ref = 1;
  mask.shape = input.shape;

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = input.shape;

  SingleOpModel model({ToString(OperationType::MUL), {}}, {input, mask},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {1, 2, 3, 4}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 4, 9, 16}));
}

TEST(MulTest, RuntimeTensorSingleChannel) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> mask;
  mask.type = DataType::FLOAT32;
  mask.ref = 1;
  mask.shape = BHWC(1, input.shape.h, input.shape.w, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = input.shape;

  SingleOpModel model({ToString(OperationType::MUL), {}}, {input, mask},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {2, 3}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {2, 4, 9, 12}));
}

TEST(MulTest, RuntimeTensorLinear) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> mask;
  mask.type = DataType::FLOAT32;
  mask.ref = 1;
  mask.shape = BHWC(1, 1, 1, input.shape.c);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = input.shape;

  SingleOpModel model({ToString(OperationType::MUL), {}}, {input, mask},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {1, 2}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {1, 4, 3, 8}));
}

TEST(MulTest, RuntimeTensorScalar) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 2, 2);

  TensorRef<BHWC> mask;
  mask.type = DataType::FLOAT32;
  mask.ref = 1;
  mask.shape = BHWC(1, 1, 1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = input.shape;

  SingleOpModel model({ToString(OperationType::MUL), {}}, {input, mask},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1, 2, 3, 4}));
  ASSERT_TRUE(model.PopulateTensor(1, {5}));
  ASSERT_OK(model.Invoke(*NewMultiplyNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), {5, 10, 15, 20}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
