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

#include "tensorflow/lite/delegates/gpu/gl/kernels/pad.h"

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

namespace {

void TestPadOperation(const HWC& prepend, const HWC& append,
                      const BHWC& output_shape, std::vector<float>&& expected) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 1, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 1;
  output.shape = output_shape;

  PadAttributes attr;
  attr.prepended = BHWC(0, prepend.h, prepend.w, prepend.c);
  attr.appended = BHWC(0, append.h, append.w, append.c);
  attr.type = PaddingContentType::ZEROS;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0}));
  ASSERT_OK(model.Invoke(*NewPadNodeShader()));
  EXPECT_THAT(model.GetOutput(0), Pointwise(FloatNear(1e-6), expected));
}

void TestPrepending(const HWC& prepend, const BHWC& output_shape,
                    std::vector<float>&& expected) {
  TestPadOperation(prepend, HWC(0, 0, 0), output_shape, std::move(expected));
}

void TestAppending(const HWC& append, const BHWC& output_shape,
                   std::vector<float>&& expected) {
  TestPadOperation(HWC(0, 0, 0), append, output_shape, std::move(expected));
}

}  // namespace

TEST(PadTest, PrependH) {
  TestPrepending(/*prepend=*/HWC(1, 0, 0),
                 /*output_shape=*/BHWC(1, 2, 1, 1), /*expected=*/{0, 1});
}

TEST(PadTest, PrependW) {
  TestPrepending(/*prepend=*/HWC(0, 1, 0), /*output_shape=*/BHWC(1, 1, 2, 1),
                 /*expected=*/{0, 1});
}

TEST(PadTest, PrependC) {
  TestPrepending(/*prepend=*/HWC(0, 0, 1), /*output_shape=*/BHWC(1, 1, 1, 2),
                 /*expected=*/{0, 1});
}

TEST(PadTest, PrependHWC) {
  TestPrepending(/*prepend=*/HWC(1, 1, 1), /*output_shape=*/BHWC(1, 2, 2, 2),
                 /*expected=*/{0, 0, 0, 0, 0, 0, 0, 1});
}

TEST(PadTest, AppendH) {
  TestAppending(/*append=*/HWC(1, 0, 0), /*output_shape=*/BHWC(1, 2, 1, 1),
                /*expected=*/{1, 0});
}

TEST(PadTest, AppendW) {
  TestAppending(/*append=*/HWC(0, 1, 0), /*output_shape=*/BHWC(1, 1, 2, 1),
                /*expected=*/{1, 0});
}

TEST(PadTest, AppendC) {
  TestAppending(/*append=*/HWC(0, 0, 1), /*output_shape=*/BHWC(1, 1, 1, 2),
                /*expected=*/{1, 0});
}

TEST(PadTest, AppendHWC) {
  TestAppending(/*append=*/HWC(1, 1, 1), /*output_shape=*/BHWC(1, 2, 2, 2),
                /*expected=*/{1, 0, 0, 0, 0, 0, 0, 0});
}

TEST(PadTest, PrependHWCAppendHWC) {
  TestPadOperation(/*prepend=*/HWC(1, 1, 1), /*append=*/HWC(1, 1, 1),
                   /*output_shape=*/BHWC(1, 3, 3, 3),
                   /*expected=*/{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
}

TEST(MirrorPadTest, Smoke) {
  TensorRef<BHWC> input;
  input.type = DataType::FLOAT32;
  input.ref = 0;
  input.shape = BHWC(1, 1, 3, 1);

  TensorRef<BHWC> output;
  output.type = DataType::FLOAT32;
  output.ref = 2;
  output.shape = BHWC(1, 1, 7, 1);

  PadAttributes attr;
  attr.prepended = BHWC(0, 0, 2, 0);
  attr.appended = BHWC(0, 0, 2, 0);
  attr.type = PaddingContentType::REFLECT;

  SingleOpModel model({ToString(OperationType::PAD), attr}, {input}, {output});
  ASSERT_TRUE(model.PopulateTensor(0, {1.0, 2.0, 3.0}));
  ASSERT_OK(model.Invoke(*NewPadNodeShader()));
  EXPECT_THAT(model.GetOutput(0),
              Pointwise(FloatNear(1e-6), {3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0}));
}

}  // namespace
}  // namespace gl
}  // namespace gpu
}  // namespace tflite
