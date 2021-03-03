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

#include "tensorflow/compiler/xla/service/gpu/cudnn_pad_for_convolutions.h"

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::_;

class CudnnPadForConvolutionsTest : public HloTestBase {};

TEST_F(CudnnPadForConvolutionsTest, PadF16ForwardConvInputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,41] parameter(0)
    filter = f16[2,2,41,40] parameter(1)
    ROOT result = (f16[10,20,30,40], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();

  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(root, op::CustomCall(kCudnnConvForwardCallTarget,
                                   op::Pad(op::Parameter(0), _),
                                   op::Pad(op::Parameter(1), _)));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),
                               ShapeUtil::MakeShape(F16, {10, 20, 30, 48})));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->shape(),
                               ShapeUtil::MakeShape(F16, {2, 2, 48, 40})));
}

TEST_F(CudnnPadForConvolutionsTest, PadF16BackwardInputConvOutputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    output = f16[10,20,30,41] parameter(0)
    filter = f16[2,2,40,41] parameter(1)
    ROOT result = (f16[10,20,30,40], u8[0]) custom-call(output, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convBackwardInput"
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::CustomCall(kCudnnConvBackwardInputCallTarget,
                                   op::Pad(op::Parameter(0), _),
                                   op::Pad(op::Parameter(1), _)));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),
                               ShapeUtil::MakeShape(F16, {10, 20, 30, 48})));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->shape(),
                               ShapeUtil::MakeShape(F16, {2, 2, 40, 48})));
}

TEST_F(CudnnPadForConvolutionsTest, PadF16ForwardConvOutputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,40] parameter(0)
    filter = f16[2,2,40,41] parameter(1)
    ROOT result = (f16[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Slice(op::GetTupleElement(op::CustomCall(
                                  kCudnnConvForwardCallTarget, op::Parameter(0),
                                  op::Pad(op::Parameter(1), _)))),
                              _));
}

TEST_F(CudnnPadForConvolutionsTest, PadF16BackwardInputConvInputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    output = f16[10,20,30,40] parameter(0)
    filter = f16[2,2,41,40] parameter(1)
    result = (f16[10,20,30,41], u8[0]) custom-call(output, filter),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardInput"
    ROOT gte = f16[10,20,30,41] get-tuple-element(result), index=0
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::Tuple(
                        op::Slice(op::GetTupleElement(op::CustomCall(
                            kCudnnConvBackwardInputCallTarget, op::Parameter(0),
                            op::Pad(op::Parameter(1), _)))),
                        _)));
}

TEST_F(CudnnPadForConvolutionsTest, PadF16BackwardFilterConvInputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,41] parameter(0)
    output = f16[10,20,30,40] parameter(1)
    result = (f16[2,2,41,40], u8[0]) custom-call(input, output),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardFilter"
    ROOT gte = f16[2,2,41,40] get-tuple-element(result), index=0
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::Tuple(
                        op::Slice(op::GetTupleElement(op::CustomCall(
                            kCudnnConvBackwardFilterCallTarget,
                            op::Pad(op::Parameter(0), _), op::Parameter(1)))),
                        _)));
}

TEST_F(CudnnPadForConvolutionsTest, PadF16BackwardFilterConvOutputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,40] parameter(0)
    output = f16[10,20,30,41] parameter(1)
    result = (f16[2,2,40,41], u8[0]) custom-call(input, output),
              window={size=2x2}, dim_labels=b01f_01io->b01f,
              custom_call_target="__cudnn$convBackwardFilter"
    ROOT gte = f16[2,2,40,41] get-tuple-element(result), index=0
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::GetTupleElement(op::Tuple(
                        op::Slice(op::GetTupleElement(op::CustomCall(
                            kCudnnConvBackwardFilterCallTarget,
                            op::Parameter(0), op::Pad(op::Parameter(1), _)))),
                        _)));
}

TEST_F(CudnnPadForConvolutionsTest, PadInputFeatures3To4) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = f16[10,20,30,3] parameter(0)
    filter = f16[2,2,3,32] parameter(1)
    ROOT result = (f16[10,20,30,32], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();

  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(root, op::CustomCall(kCudnnConvForwardCallTarget,
                                   op::Pad(op::Parameter(0), _),
                                   op::Pad(op::Parameter(1), _)));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),
                               ShapeUtil::MakeShape(F16, {10, 20, 30, 4})));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->shape(),
                               ShapeUtil::MakeShape(F16, {2, 2, 4, 32})));
}

TEST_F(CudnnPadForConvolutionsTest, PadIntForwardConvInputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,41] parameter(0)
    filter = s8[2,2,41,40] parameter(1)
    ROOT result = (f32[10,20,30,40], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();

  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(root, op::CustomCall(kCudnnConvForwardCallTarget,
                                   op::Pad(op::Parameter(0), _),
                                   op::Pad(op::Parameter(1), _)));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(0)->shape(),

                               ShapeUtil::MakeShape(S8, {10, 20, 30, 44})));
  EXPECT_TRUE(ShapeUtil::Equal(root->operand(1)->shape(),
                               ShapeUtil::MakeShape(S8, {2, 2, 44, 40})));
}

TEST_F(CudnnPadForConvolutionsTest, PadIntForwardConvOutputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,40] parameter(0)
    filter = s8[2,2,40,41] parameter(1)
    ROOT result = (f32[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Slice(op::GetTupleElement(op::CustomCall(
                                  kCudnnConvForwardCallTarget, op::Parameter(0),
                                  op::Pad(op::Parameter(1), _)))),
                              _));
}

TEST_F(CudnnPadForConvolutionsTest,
       PadIntFusedForwardConvInputAndOutputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule Test

    ENTRY %Test (input: s8[1,3,3,2], filter: s8[3,3,2,5], side_input: s8[1,3,3,5], bias: s8[5]) -> f32[1,3,3,5] {
    %input = s8[1,3,3,2]{3,2,1,0} parameter(0)
    %filter = s8[3,3,2,5]{3,2,1,0} parameter(1)
    %bias = s8[5]{0} parameter(3)
    %convert = f32[5]{0} convert(s8[5]{0} %bias)
    %side_input = f32[1,3,3,5]{3,2,1,0} parameter(2)
    %custom-call.1 = (f32[1,3,3,5]{3,2,1,0}, u8[0]{0}) custom-call(s8[1,3,3,2]{3,2,1,0} %input, s8[3,3,2,5]{3,2,1,0} %filter, f32[5]{0} %convert, f32[1,3,3,5]{3,2,1,0} %side_input), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config="{\"activationMode\":\"2\",\"convResultScale\":1,\"sideInputScale\":1}"
    ROOT %get-tuple-element.1 = f32[1,3,3,5]{3,2,1,0} get-tuple-element((f32[1,3,3,5]{3,2,1,0}, u8[0]{0}) %custom-call.1), index=0
    })")
                    .ValueOrDie();
  EXPECT_TRUE(CudnnPadForConvolutions(true).Run(module.get()).ValueOrDie());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, op::GetTupleElement(op::Tuple(
                op::Slice(op::GetTupleElement(op::CustomCall(
                    kCudnnConvBiasActivationForwardCallTarget,
                    op::Pad(op::Parameter(0), _), op::Pad(op::Parameter(1), _),
                    op::Pad(op::Convert(op::Parameter(3)), _),
                    op::Pad(op::Parameter(2), _)))),
                _)));
}
}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
