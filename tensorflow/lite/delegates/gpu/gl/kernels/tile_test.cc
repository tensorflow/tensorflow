/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/tile.h"

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

TEST(TileTest, ChannelsTiling) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 1, 3), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 1, 6), .ref = 1};
  SingleOpModel model({ToString(OperationType::TILE)}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  ASSERT_OK(model.Invoke(*NewTileNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(TileTest, WidthTiling) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 1, 2, 3), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 1, 4, 3), .ref = 1};
  SingleOpModel model({ToString(OperationType::TILE)}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  ASSERT_OK(model.Invoke(*NewTileNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(TileTest, HeightTiling) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 1, 3), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 4, 1, 3), .ref = 1};
  SingleOpModel model({ToString(OperationType::TILE)}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  ASSERT_OK(model.Invoke(*NewTileNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                          1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
}

TEST(TileTest, HWCTiling) {
  const TensorRef<BHWC> input = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 2, 2, 3), .ref = 0};
  const TensorRef<BHWC> output = {
      .type = DataType::FLOAT32, .shape = BHWC(1, 4, 4, 6), .ref = 1};
  SingleOpModel model({ToString(OperationType::TILE)}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
                                       8.0f, 9.0f, 10.0f, 11.0f, 12.0f}));
  ASSERT_OK(model.Invoke(*NewTileNodeShader()));
  EXPECT_THAT(
      model.GetOutput(0),
      Pointwise(
          FloatNear(1e-6),
          {1.0f,  2.0f,  3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  4.0f,
           5.0f,  6.0f,  1.0f,  2.0f,  3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
           6.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  7.0f,  8.0f,  9.0f,
           10.0f, 11.0f, 12.0f, 10.0f, 11.0f, 12.0f, 7.0f,  8.0f,  9.0f,  7.0f,
           8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 10.0f, 11.0f, 12.0f, 1.0f,  2.0f,
           3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  4.0f,  5.0f,  6.0f,
           1.0f,  2.0f,  3.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  4.0f,
           5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
           12.0f, 10.0f, 11.0f, 12.0f, 7.0f,  8.0f,  9.0f,  7.0f,  8.0f,  9.0f,
           10.0f, 11.0f, 12.0f, 10.0f, 11.0f, 12.0f}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
