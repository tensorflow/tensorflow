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

class ElementwiseTest : public ::testing::Test {
 public:
  ElementwiseTest() = default;
  ~ElementwiseTest() override = default;

  TensorRefFloat32 GetTensorRef(int ref) {
    TensorRefFloat32 tensor_ref;
    tensor_ref.type = DataType::FLOAT32;
    tensor_ref.ref = ref;
    tensor_ref.shape = BHWC(1, 2, 2, 1);
    return tensor_ref;
  }
};

TEST_F(ElementwiseTest, Abs) {
  OperationType op_type = OperationType::ABS;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.2, 2.0, 4.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 6.2, 2.0, 4.0}));
}

TEST_F(ElementwiseTest, Sin) {
  OperationType op_type = OperationType::SIN;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 3.1415926, -3.1415926, 1.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 0.0, 0.0, 0.841471}));
}

TEST_F(ElementwiseTest, Cos) {
  OperationType op_type = OperationType::COS;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 3.1415926, -3.1415926, 1}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, -1.0, -1.0, 0.540302}));
}

TEST_F(ElementwiseTest, Log) {
  OperationType op_type = OperationType::LOG;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 3.1415926, 1.0, 1.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.14473, 0.0, 0.0}));
}

TEST_F(ElementwiseTest, Sqrt) {
  OperationType op_type = OperationType::SQRT;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, 1.0, 2.0, 4.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, 1.0, 1.414213, 2.0}));
}

TEST_F(ElementwiseTest, Rsqrt) {
  OperationType op_type = OperationType::RSQRT;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 4.0, 9.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 0.707106, 0.5, 0.333333}));
}

TEST_F(ElementwiseTest, Square) {
  OperationType op_type = OperationType::SQUARE;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 0.5, -3.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 4.0, 0.25, 9.0}));
}

TEST_F(ElementwiseTest, Sigmoid) {
  OperationType op_type = OperationType::SIGMOID;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.0, 2.0, 4.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.5, 0.002473, 0.880797, 0.982014}));
}

TEST_F(ElementwiseTest, Tanh) {
  OperationType op_type = OperationType::TANH;
  SingleOpModel model({ToString(op_type), {}}, {GetTensorRef(0)},
                      {GetTensorRef(1)});
  ASSERT_TRUE(model.PopulateTensor(0, {0.0, -6.0, 2.0, 4.0}));
  ASSERT_TRUE(model.Invoke(*NewElementwiseNodeShader(op_type)));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {0.0, -0.999987, 0.964027, 0.999329}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
