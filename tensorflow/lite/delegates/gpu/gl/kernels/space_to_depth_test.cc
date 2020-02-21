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

#include "tensorflow/lite/delegates/gpu/gl/kernels/space_to_depth.h"

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

TEST(SpaceToDepthTest, TensorShape1x2x2x1BlockSize2) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 1), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 1, 1, 4), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0f, 2.0f, 3.0f, 4.0f}));
  ASSERT_OK(model.Invoke(*NewSpaceToDepthNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(SpaceToDepthTest, TensorShape1x2x2x2BlockSize2) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 2), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 1, 1, 8), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(
      0, {1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f}));
  ASSERT_OK(model.Invoke(*NewSpaceToDepthNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6),
                        {1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f}));
}

TEST(SpaceToDepthTest, TensorShape1x2x2x3BlockSize2) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 3), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 1, 1, 12), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0f, 2.0f, 3.0f,  //
                                       4.0f, 5.0f, 6.0f,  //
                                       7.0f, 8.0f, 9.0f,  //
                                       10.0f, 11.0f, 12.0f}));
  ASSERT_OK(model.Invoke(*NewSpaceToDepthNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(FloatNear(1e-6), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,  //
                                  7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
}

TEST(SpaceToDepthTest, TensorShape1x4x4x1BlockSize2) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 4, 4, 1), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 4), .ref = 1};
  const SpaceToDepthAttributes attr = {.block_size = 2};
  SingleOpModel model({ToString(OperationType::SPACE_TO_DEPTH), attr}, {input},
                      {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 5.0, 6.0,     //
                                       3.0, 4.0, 7.0, 8.0,     //
                                       9.0, 10.0, 13.0, 14.0,  //
                                       11.0, 12.0, 15.0, 16.0}));
  ASSERT_OK(model.Invoke(*NewSpaceToDepthNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0, 2.0, 3.0, 4.0,     //
                                          5.0, 6.0, 7.0, 8.0,     //
                                          9.0, 10.0, 11.0, 12.0,  //
                                          13.0, 14.0, 15.0, 16.0}));
}
}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
