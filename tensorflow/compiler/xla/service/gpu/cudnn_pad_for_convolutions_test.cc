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

#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

namespace m = xla::match;

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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();

  SCOPED_TRACE(module->ToString());

  EXPECT_THAT(
      root,
      GmockMatch(m::CustomCall(
          kCudnnConvForwardCallTarget,
          m::Pad(m::Parameter(0), m::Op()).WithShape(F16, {10, 20, 30, 48}),
          m::Pad(m::Parameter(1), m::Op()).WithShape(F16, {2, 2, 48, 40}))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::CustomCall(
          kCudnnConvBackwardInputCallTarget,
          m::Pad(m::Parameter(0), m::Op()).WithShape(F16, {10, 20, 30, 48}),
          m::Pad(m::Parameter(1), m::Op()).WithShape(F16, {2, 2, 40, 48}))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(
                        m::Slice(m::GetTupleElement(m::CustomCall(
                            kCudnnConvForwardCallTarget, m::Parameter(0),
                            m::Pad(m::Parameter(1), m::Op())))),
                        m::Op())));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(m::Tuple(
                        m::Slice(m::GetTupleElement(m::CustomCall(
                            kCudnnConvBackwardInputCallTarget, m::Parameter(0),
                            m::Pad(m::Parameter(1), m::Op())))),
                        m::Op()))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              GmockMatch(m::GetTupleElement(m::Tuple(
                  m::Slice(m::GetTupleElement(m::CustomCall(
                      kCudnnConvBackwardFilterCallTarget,
                      m::Pad(m::Parameter(0), m::Op()), m::Parameter(1)))),
                  m::Op()))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(m::Tuple(
                        m::Slice(m::GetTupleElement(m::CustomCall(
                            kCudnnConvBackwardFilterCallTarget, m::Parameter(0),
                            m::Pad(m::Parameter(1), m::Op())))),
                        m::Op()))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();

  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      root,
      GmockMatch(m::CustomCall(
          kCudnnConvForwardCallTarget,
          m::Pad(m::Parameter(0), m::Op()).WithShape(F16, {10, 20, 30, 4}),
          m::Pad(m::Parameter(1), m::Op()).WithShape(F16, {2, 2, 4, 32}))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();

  SCOPED_TRACE(module->ToString());
  EXPECT_THAT(
      root,
      GmockMatch(m::CustomCall(
          kCudnnConvForwardCallTarget,
          m::Pad(m::Parameter(0), m::Op()).WithShape(S8, {10, 20, 30, 44}),
          m::Pad(m::Parameter(1), m::Op()).WithShape(S8, {2, 2, 44, 40}))));
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
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(
                        m::Slice(m::GetTupleElement(m::CustomCall(
                            kCudnnConvForwardCallTarget, m::Parameter(0),
                            m::Pad(m::Parameter(1), m::Op())))),
                        m::Op())));
}

TEST_F(CudnnPadForConvolutionsTest, PadInt8To32OnSm75) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,40] parameter(0)
    filter = s8[2,2,40,41] parameter(1)
    ROOT result = (s8[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 5}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Slice(m::GetTupleElement(m::CustomCall(
              kCudnnConvForwardCallTarget,
              m::Pad(m::Parameter(0), m::Op()).WithShape(S8, {10, 20, 30, 64}),
              m::Pad(m::Parameter(1), m::Op()).WithShape(S8, {2, 2, 64, 64})))),
          m::Op())));
}

TEST_F(CudnnPadForConvolutionsTest, NoPadInt8To32OnSm70) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,40] parameter(0)
    filter = s8[2,2,40,41] parameter(1)
    ROOT result = (s8[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Slice(m::GetTupleElement(m::CustomCall(
              kCudnnConvForwardCallTarget, m::Parameter(0),
              m::Pad(m::Parameter(1), m::Op()).WithShape(S8, {2, 2, 40, 44})))),
          m::Op())));
}

TEST_F(CudnnPadForConvolutionsTest, NoPadInt8To32FloatOutputSm75) {
  // This test checks that the padding pass correctly calls
  // CudnnSupportsOptimizedIntegerConvolution() which should reject this
  // convolution because its output type is f32. It should be padded to int8x4
  // because that supports float outputs.
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,38] parameter(0)
    filter = s8[2,2,38,41] parameter(1)
    ROOT result = (f32[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 5}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Slice(m::GetTupleElement(m::CustomCall(
              kCudnnConvForwardCallTarget,
              m::Pad(m::Parameter(0), m::Op()).WithShape(S8, {10, 20, 30, 40}),
              m::Pad(m::Parameter(1), m::Op()).WithShape(S8, {2, 2, 40, 44})))),
          m::Op())));
}

TEST_F(CudnnPadForConvolutionsTest, NoPadInt8UnsupportedFilterTypeOutputSm75) {
  // This test checks that the padding pass correctly calls
  // CudnnSupportsOptimizedIntegerConvolution() which should reject this
  // convolution because kernel type is f32.
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,38] parameter(0)
    filter = f32[2,2,38,41] parameter(1)
    ROOT result = (s8[10,20,30,41], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f_01io->b01f,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  EXPECT_FALSE(CudnnPadForConvolutions({7, 5}).Run(module.get()).value());
}

TEST_F(CudnnPadForConvolutionsTest, NoPadToInt8x32ExcessiveBlowup) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[128,4,48,48] parameter(0)
    filter = s8[64,4,3,3] parameter(1)
    ROOT result = (f32[128,64,48,48], u8[0]) custom-call(input, filter),
                  window={size=3x3}, dim_labels=bf01_io01->bf01,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  EXPECT_FALSE(CudnnPadForConvolutions({7, 5}).Run(module.get()).value());
}

TEST_F(CudnnPadForConvolutionsTest, PadInt8x4To32) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,41,4] parameter(0)
    filter = s8[2,2,41,4,168] parameter(1)
    ROOT result = (s8[10,20,30,42,4], u8[0]) custom-call(input, filter),
                  window={size=2x2}, dim_labels=b01f?_01i?o->b01f?,
                  custom_call_target="__cudnn$convForward"
  })")
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 5}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Slice(m::GetTupleElement(
                       m::CustomCall(kCudnnConvForwardCallTarget,
                                     m::Pad(m::Parameter(0), m::Op())
                                         .WithShape(S8, {10, 20, 30, 48, 4}),
                                     m::Pad(m::Parameter(1), m::Op())
                                         .WithShape(S8, {2, 2, 48, 4, 192})))
                       .WithShape(S8, {10, 20, 30, 48, 4})),
          m::Op())));
}

TEST_F(CudnnPadForConvolutionsTest, PadInt8x4To32BiasActivation) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule TestModule

  ENTRY TestComputation {
    input = s8[10,20,30,41,4] parameter(0)
    filter = s8[2,2,41,4,168] parameter(1)
    bias = f32[10] parameter(2)
    side_input = s8[10,20,30,42,4] parameter(3)
    ROOT result = (s8[10,20,30,42,4], u8[0]) custom-call(input, filter, bias, side_input),
                  window={size=2x2}, dim_labels=b01f?_01i?o->b01f?,
                  custom_call_target="__cudnn$convBiasActivationForward"
  })")
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 5}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root,
      GmockMatch(m::Tuple(
          m::Slice(
              m::GetTupleElement(
                  m::CustomCall(
                      kCudnnConvBiasActivationForwardCallTarget,
                      m::Pad(m::Parameter(0), m::Op())
                          .WithShape(S8, {10, 20, 30, 48, 4}),
                      m::Pad(m::Parameter(1), m::Op())
                          .WithShape(S8, {2, 2, 48, 4, 192}),
                      m::Pad(m::Parameter(2), m::Op()).WithShape(F32, {32}),
                      m::Pad(m::Parameter(3), m::Op())
                          .WithShape(S8, {10, 20, 30, 48, 4})))
                  .WithShape(S8, {10, 20, 30, 48, 4})),
          m::Op())));
}

TEST_F(CudnnPadForConvolutionsTest,
       PadIntFusedForwardConvInputAndOutputChannels) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule Test

    ENTRY %Test (input: s8[1,3,3,2], filter: s8[3,3,2,5], side_input: s8[1,3,3,5], bias: s8[5]) -> f32[1,3,3,5] {
    %input = s8[1,3,3,3]{3,2,1,0} parameter(0)
    %filter = s8[3,3,2,5]{3,2,1,0} parameter(1)
    %bias = s8[5]{0} parameter(3)
    %convert = f32[5]{0} convert(s8[5]{0} %bias)
    %side_input = f32[1,3,3,5]{3,2,1,0} parameter(2)
    %custom-call.1 = (f32[1,3,3,5]{3,2,1,0}, u8[0]{0}) custom-call(s8[1,3,3,3]{3,2,1,0} %input, s8[3,3,2,5]{3,2,1,0} %filter, f32[5]{0} %convert, f32[1,3,3,5]{3,2,1,0} %side_input), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, custom_call_target="__cudnn$convBiasActivationForward", backend_config="{\"activationMode\":\"2\",\"convResultScale\":1,\"sideInputScale\":1}"
    ROOT %get-tuple-element.1 = f32[1,3,3,5]{3,2,1,0} get-tuple-element((f32[1,3,3,5]{3,2,1,0}, u8[0]{0}) %custom-call.1), index=0
    })")
                    .value();
  EXPECT_TRUE(CudnnPadForConvolutions({7, 0}).Run(module.get()).value());
  auto* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::GetTupleElement(m::Tuple(
                        m::Slice(m::GetTupleElement(m::CustomCall(
                            kCudnnConvBiasActivationForwardCallTarget,
                            m::Pad(m::Parameter(0), m::Op()),
                            m::Pad(m::Parameter(1), m::Op()),
                            m::Pad(m::Convert(m::Parameter(3)), m::Op()),
                            m::Pad(m::Parameter(2), m::Op())))),
                        m::Op()))));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
