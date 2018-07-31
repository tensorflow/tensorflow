/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/pad_for_tensor_cores.h"

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_verified_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;

using PadForTensorCoresTest = HloVerifiedTestBase;

TEST_F(PadForTensorCoresTest, PadF16ForwardConvInputChannels) {
  ParseAndVerifyModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,41] parameter(0)
    filter = f16[2,2,41,40] parameter(1)
    ROOT result = (f16[10,20,30,40], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })");
  EXPECT_TRUE(PadForTensorCores().Run(&module()).ValueOrDie());
  auto* root = module().entry_computation()->root_instruction();

  SCOPED_TRACE(module().ToString());
  EXPECT_THAT(root, op::CustomCall(kCudnnConvForwardCallTarget,
                                   op::Pad(op::Parameter(0), _),
                                   op::Pad(op::Parameter(1), _)));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),
                               ShapeUtil::MakeShape(F16, {10, 20, 30, 48})));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->shape(),
                               ShapeUtil::MakeShape(F16, {2, 2, 48, 40})));
}

TEST_F(PadForTensorCoresTest, PadF16BackwardInputConvOutputChannels) {
  ParseAndVerifyModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    output = f16[10,20,30,41] parameter(0)
    filter = f16[2,2,40,41] parameter(1)
    ROOT result = (f16[10,20,30,40], u8[0]) custom-call(output, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convBackwardInput"
  })");
  EXPECT_TRUE(PadForTensorCores().Run(&module()).ValueOrDie());
  auto* root = module().entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CustomCall(kCudnnConvBackwardInputCallTarget,
                                   op::Pad(op::Parameter(0), _),
                                   op::Pad(op::Parameter(1), _)));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),
                               ShapeUtil::MakeShape(F16, {10, 20, 30, 48})));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->shape(),
                               ShapeUtil::MakeShape(F16, {2, 2, 40, 48})));
}

TEST_F(PadForTensorCoresTest, PadF16ForwardConvOutputChannels) {
  ParseAndVerifyModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,40] parameter(0)
    filter = f16[2,2,40,41] parameter(1)
    ROOT result = (f16[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })");
  EXPECT_TRUE(PadForTensorCores().Run(&module()).ValueOrDie());
  auto* root = module().entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Slice(op::GetTupleElement(op::CustomCall(
                                  kCudnnConvForwardCallTarget, op::Parameter(0),
                                  op::Pad(op::Parameter(1), _)))),
                              _));
}

TEST_F(PadForTensorCoresTest, PadF16BackwardInputConvInputChannels) {
  ParseAndVerifyModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    output = f16[10,20,30,40] parameter(0)
    filter = f16[2,2,41,40] parameter(1)
    result = (f16[10,20,30,41], u8[0]) custom-call(output, filter),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardInput"
    ROOT gte = f16[10,20,30,41] get-tuple-element(result), index=0
  })");
  EXPECT_TRUE(PadForTensorCores().Run(&module()).ValueOrDie());
  auto* root = module().entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::Tuple(
                        op::Slice(op::GetTupleElement(op::CustomCall(
                            kCudnnConvBackwardInputCallTarget, op::Parameter(0),
                            op::Pad(op::Parameter(1), _)))),
                        _)));
}

TEST_F(PadForTensorCoresTest, PadF16BackwardFilterConvInputChannels) {
  ParseAndVerifyModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,41] parameter(0)
    output = f16[10,20,30,40] parameter(1)
    result = (f16[2,2,41,40], u8[0]) custom-call(input, output),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardFilter"
    ROOT gte = f16[2,2,41,40] get-tuple-element(result), index=0
  })");
  EXPECT_TRUE(PadForTensorCores().Run(&module()).ValueOrDie());
  auto* root = module().entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::Tuple(
                        op::Slice(op::GetTupleElement(op::CustomCall(
                            kCudnnConvBackwardFilterCallTarget,
                            op::Pad(op::Parameter(0), _), op::Parameter(1)))),
                        _)));
}

TEST_F(PadForTensorCoresTest, PadF16BackwardFilterConvOutputChannels) {
  ParseAndVerifyModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,40] parameter(0)
    output = f16[10,20,30,41] parameter(1)
    result = (f16[2,2,40,41], u8[0]) custom-call(input, output),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardFilter"
    ROOT gte = f16[2,2,40,41] get-tuple-element(result), index=0
  })");
  EXPECT_TRUE(PadForTensorCores().Run(&module()).ValueOrDie());
  auto* root = module().entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::Tuple(
                        op::Slice(op::GetTupleElement(op::CustomCall(
                            kCudnnConvBackwardFilterCallTarget,
                            op::Parameter(0), op::Pad(op::Parameter(1), _)))),
                        _)));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
