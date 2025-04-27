/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/copy_fusion.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla {
namespace gpu {

namespace m = ::xla::match;

auto MakeDeviceDescriptor() {
  stream_executor::DeviceDescription device_description{
      stream_executor::GpuDeviceInfoProto{}};
  device_description.set_threads_per_warp(32);
  return device_description;
}

class CopyFusionTest : public HloHardwareIndependentTestBase {
 public:
  CopyFusionTest()
      : device_description_(MakeDeviceDescriptor()), cf_(device_description_) {}
  const stream_executor::DeviceDescription device_description_;
  CopyFusion cf_;
};

const char kModulePrefix[] = R"(
    HloModule test_module

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }
    scalar_mul_computation {
      scalar_lhs.1 = f32[] parameter(0)
      scalar_rhs.1 = f32[] parameter(1)
      ROOT mul.1 = f32[] multiply(scalar_lhs.1, scalar_rhs.1)
    })";

TEST_F(CopyFusionTest, CopyFusionTransposeOfBroadcastedConstantTwoCopies) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      two = f32[] constant(2.0)
      broadcast = f32[16,32]{1,0} broadcast(two), dimensions={}
      s.1 = f32[16,32]{1,0} sqrt(broadcast)
      ROOT c.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
    }

    ENTRY main {
      fusion = f32[32,16]{1,0} fusion(), kind=kInput, calls=fused_computation
      copy.1 = f32[32,16]{1,0} copy(fusion)
      copy.2 = f32[32,16]{1,0} copy(fusion)
      ROOT t = (f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(copy.2, copy.1)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Transpose(), m::Copy(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionTransposeTwoCopies) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      param_0.1 = f32[16,32]{1,0} parameter(0)
      s.1 = f32[16,32]{1,0} sqrt(param_0.1)
      ROOT c.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
    }

    ENTRY main {
      p = f32[16,32]{1,0} parameter(0)
      fusion = f32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
      copy.1 = f32[32,16]{1,0} copy(fusion)
      copy.2 = f32[32,16]{1,0} copy(fusion)
      ROOT t = (f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(copy.2, copy.1)
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionNegateAndTwoCopies) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      ROOT neg = f32[128,512,28,28]{3,2,1,0} negate(mul)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = f32[128,512,28,28]{3,2,1,0} fusion(p0), kind=kInput, calls=fused_computation
      copy.1 = f32[128,512,28,28]{3,2,1,0} copy(fusion)
      copy.2 = f32[128,512,28,28]{3,2,1,0} copy(fusion)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(copy.1, copy.2)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Negate(), m::Copy(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionShouldNotRunWithReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      fusion = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation
      copy.1 = f32[512]{0} copy(fusion)
      copy.2 = f32[512]{0} copy(fusion)
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(copy.1, copy.2)
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionShouldRunWithUncopiedReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      two = f32[] constant(2.0)
      broadcast = f32[128,512,28,28]{3,2,1,0} broadcast(two)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(broadcast, broadcast)
      const = f32[] constant(0.0)
      reduce = f32[512]{0} reduce(mul, const), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT tuple = (f32[128,512,28,28]{3,2,1,0}, f32[512]{0}) tuple(mul, reduce)
    }

    ENTRY entry {
      fusion = (f32[128,512,28,28]{3,2,1,0}, f32[512]) fusion(), kind=kInput, calls=fused_computation
      gte = f32[128,512,28,28]{3,2,1,0} get-tuple-element(fusion), index=0
      gte.2 = f32[512]{0} get-tuple-element(fusion), index=1
      copy.1 = f32[128,512,28,28]{3,2,1,0} copy(gte)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[512]{0}) tuple(copy.1, gte.2)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Multiply(), m::Reduce(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionShouldNotFuseForSliceMultioutputFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1, p1)
      slice1 = f32[128,100,28,28]{3,2,1,0} slice(mul), slice={[0:128],[0:100],[0:28],[0:28]}
      slice2 = f32[128,200,28,28]{3,2,1,0} slice(mul), slice={[0:128],[50:250],[0:28],[0:28]}
      ROOT tuple = (f32[128,100,28,28]{3,2,1,0}, f32[128,200,28,28]{3,2,1,0}) tuple(slice1, slice2)
    }

    ENTRY entry {
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[128,100,28,28]{3,2,1,0}, f32[128,200,28,28]{3,2,1,0}) fusion(p1), kind=kInput, calls=fused_computation
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionShouldNotRunWithScatter) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p0 = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
      scatter_indices = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
      updates = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
      input_tensor = f32[50,49,48,47,46]{4,3,2,1,0} negate(p0)
      ROOT %scatter = f32[50,49,48,47,46]{4,3,2,1,0} scatter(input_tensor, scatter_indices, updates), update_window_dims={4,5,6,7,8}, inserted_window_dims={}, scatter_dims_to_operand_dims={0,1,2,3,4}, index_vector_dim=4, to_apply=scalar_add_computation
}

    ENTRY entry {
      param.0 = f32[50,49,48,47,46]{4,3,2,1,0} parameter(0)
      param.1 = s64[10,9,8,7,5]{4,3,2,1,0} parameter(1)
      param.2 = f32[10,9,8,7,30,29,28,27,26]{8,7,6,5,4,3,2,1,0} parameter(2)
      fusion = f32[50,49,48,47,46]{4,3,2,1,0} fusion(param.0, param.1, param.2), kind=kInput, calls=fused_computation
      ROOT copy = f32[50,49,48,47,46]{4,3,2,1,0} copy(fusion)
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionShouldNotRunOutsideEntryComputation) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
fused_computation.549 {
  param_0.8511 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} parameter(0)
  bitcast.52601 = bf16[15,1,2,48,128,2048]{5,4,3,2,1,0} bitcast(param_0.8511)
  slice = bf16[15,1,2,48,128,1]{5,4,3,2,1,0} slice(bitcast.52601), slice={[0:15:1], [0:1:1], [0:2:1], [0:48:1], [0:128:1], [0:1:1]}
  bitcast = bf16[15,1,2,48,128]{4,3,2,1,0} bitcast(slice)
  ROOT broadcast = bf16[15,1,2,48,128,2048]{5,4,3,2,1,0} broadcast(bitcast), dimensions={0,1,2,3,4}
}

condition {
  constant_6915 = s32[] constant(15)
  param.218 = (bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) parameter(0)
  get-tuple-element.3714 = s32[] get-tuple-element(param.218), index=1
  ROOT compare.1738 = pred[] compare(get-tuple-element.3714, constant_6915), direction=LT
}

body {
  tuple_param = (bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) parameter(0)
  param_0 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} get-tuple-element(tuple_param), index=0
  param_1 = s32[] get-tuple-element(tuple_param), index=1
  fusion.549 = bf16[15,1,2,48,128,2048]{5,4,3,2,1,0} fusion(param_0), kind=kLoop, calls=fused_computation.549
  bitcast = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} bitcast(fusion.549)
  copy = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} copy(bitcast)
  constant_one = s32[] constant(1)
  add = s32[] add(param_1, constant_one), control-predecessors={fusion.549}
  ROOT tuple = (bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) tuple(copy, add)
}

ENTRY main {
  param_0 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} parameter(0)
  zero = s32[] constant(0)
  copy.0 = bf16[15,1,2,2048,48,128]{3,5,4,2,1,0} copy(param_0)
  copy.1 = s32[] copy(zero)
  tuple = tuple(copy.0, copy.1)
  ROOT while = (bf16[15,1,2,2048,48,128]{3,5,4,2,1,0}, s32[]) while(tuple), condition=condition, body=body, backend_config="{\"known_trip_count\":{\"n\":\"15\"}}"
})"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionShouldNotRunWithDynamicUpdateSliceInplace) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p.0 = f16[50,96,1024]{2,1,0} parameter(0)
      p.1 = f16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = f16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    ENTRY entry {
      p0 = f16[50,96,1024]{2,1,0} parameter(0)
      p1 = f16[1,96,1024]{2,1,0} parameter(1)
      fusion = f16[50,96,1024]{2,1,0} fusion(p0, p1), kind=kInput, calls=fused_computation
      copy.1 = f16[50,96,1024]{2,1,0} copy(fusion)
      copy.2 = f16[50,96,1024]{2,1,0} copy(fusion)
      ROOT root = (f16[50,96,1024]{2,1,0}, f16[50,96,1024]{2,1,0}) tuple(copy.1, copy.2)
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionWithDynamicUpdateSliceNotInplace) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      one = f32[] constant(1.0)
      zero = f32[] constant(0.0)
      p.0 = f16[50,96,1024]{2,1,0} broadcast(one), dimensions={}
      p.1 = f16[1,96,1024]{2,1,0} broadcast(zero), dimensions={}
      c.0 = s32[3]{0} constant({0, 0, 0})
      dynamic-update-slice = f16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
      neg = f16[50,96,1024]{2,1,0} negate(dynamic-update-slice)
      ROOT tuple = (f16[50,96,1024]{2,1,0}, f16[50,96,1024]{2,1,0}) tuple(dynamic-update-slice, neg)
    }

    ENTRY entry {
      fusion = (f16[50,96,1024]{2,1,0}, f16[50,96,1024]{2,1,0}) fusion(), kind=kInput, calls=fused_computation
      gte.0 = f16[50,96,1024]{2,1,0} get-tuple-element(fusion), index=0
      gte.1 = f16[50,96,1024]{2,1,0} get-tuple-element(fusion), index=1
      bitcast = f16[1,50,96,1024]{3,2,1,0} bitcast(gte.0)
      copy = f16[1,50,96,1024]{3,2,1,0} copy(bitcast)
      ROOT root = (f16[1,50,96,1024]{3,2,1,0}, f16[50,96,1024]{2,1,0}) tuple(copy, gte.1)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  EXPECT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(m::Tuple(m::DynamicUpdateSlice(), m::Negate(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionTransposeAndThreeCopies) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      two = f32[] constant(2.0)
      param_0.1 = f32[16,32]{1,0} broadcast(two), dimensions={}
      s.1 = f32[16,32]{1,0} sqrt(param_0.1)
      ROOT c.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
    }

    ENTRY entry {
      fusion = f32[32,16]{1,0} fusion(), kind=kInput, calls=fused_computation
      copy.1 = f32[32,16]{1,0} copy(fusion)
      copy.2 = f32[32,16]{1,0} copy(fusion)
      copy.3 = f32[32,16]{1,0} copy(fusion)
      ROOT root = (f32[32,16]{1,0}, f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(copy.1, copy.2, copy.3)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                  m::GetTupleElement(), m::GetTupleElement())));
  EXPECT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(m::Tuple(m::Transpose(), m::Copy(), m::Copy(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionRunWithOnlyOneCopy) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      ROOT neg = f32[128,512,28,28]{3,2,1,0} negate(mul)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = f32[128,512,28,28]{3,2,1,0} fusion(p0), kind=kInput, calls=fused_computation
      ROOT copy.1 = f32[128,512,28,28]{3,2,1,0} copy(fusion)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::GetTupleElement(m::Fusion(&fusion))));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Negate(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionNegateAndTwoCopiesAndTransposeCopy) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      ROOT neg = f32[128,512,28,28]{3,2,1,0} negate(mul)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = f32[128,512,28,28]{3,2,1,0} fusion(p0), kind=kInput, calls=fused_computation
      copy.1 = f32[128,512,28,28]{3,2,1,0} copy(fusion)
      transpose = f32[128,512,28,28]{2,3,0,1} copy(fusion)
      bitcast = f32[512,128,28,28]{3,2,1,0} bitcast(transpose)
      copy.2 = f32[128,512,28,28]{3,2,1,0} copy(fusion)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[512,128,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(copy.1, bitcast, copy.2)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::Bitcast(), m::GetTupleElement())));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Negate(), m::Copy(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionRunWithOnlyOneNonTransposeCopy) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      ROOT neg = f32[128,512,28,28]{3,2,1,0} negate(mul)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = f32[128,512,28,28]{3,2,1,0} fusion(p0), kind=kInput, calls=fused_computation
      copy.1 = f32[128,512,28,28]{3,2,1,0} copy(fusion)
      transpose.1 = f32[128,512,28,28]{2,3,0,1} copy(fusion)
      bitcast.1 = f32[512,128,28,28]{3,2,1,0} bitcast(transpose.1)
      transpose.2 = f32[128,512,28,28]{2,3,0,1} copy(fusion)
      bitcast.2 = f32[512,128,28,28]{3,2,1,0} bitcast(transpose.2)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[512,128,28,28]{3,2,1,0}, f32[512,128,28,28]{3,2,1,0}) tuple(copy.1, bitcast.1, bitcast.2)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::Bitcast(), m::Bitcast())));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Negate(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionSkipTupleCopies) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      neg.1 = f32[128,512,28,28]{3,2,1,0} negate(mul)
      neg.2 = f32[128,512,28,28]{3,2,1,0} negate(mul)
      ROOT tuple = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(neg.1, neg.2)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kInput, calls=fused_computation
      copy.1 = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) copy(fusion)
      copy.2 = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) copy(fusion)
      ROOT root = ((f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}),(f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0})) tuple(copy.1, copy.2)
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionTupleAndGetTuple) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      neg.1 = f32[128,512,28,28]{3,2,1,0} negate(mul)
      neg.2 = f32[128,512,28,28]{3,2,1,0} negate(mul)
      ROOT tuple = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(neg.1, neg.2)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kInput, calls=fused_computation
      gte.1 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(fusion), index=0
      gte.2 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(fusion), index=1
      copy.1 = f32[128,512,28,28]{3,2,1,0} copy(gte.1)
      copy.2 = f32[128,512,28,28]{3,2,1,0} copy(gte.2)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(copy.1, copy.2)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Tuple(m::GetTupleElement(m::Fusion(&fusion)),
                                        m::GetTupleElement())));
  EXPECT_THAT(
      fusion->fused_expression_root(),
      GmockMatch(m::Tuple(m::Negate(), m::Negate(), m::Copy(), m::Copy())));
}

TEST_F(CopyFusionTest, CopyFusionWithFusionReturningTupleAndOtherUser) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      neg.1 = f32[128,512,28,28]{3,2,1,0} negate(mul)
      neg.2 = f32[128,512,28,28]{3,2,1,0} negate(mul)
      ROOT tuple = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(neg.1, neg.2)
    }

    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      fusion = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kInput, calls=fused_computation
      gte.1 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(fusion), index=0
      gte.2 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(fusion), index=1
      copy.1 = f32[128,512,28,28]{3,2,1,0} copy(gte.1)
      copy.2 = f32[128,512,28,28]{3,2,1,0} copy(gte.2)
      transpose = f32[128,512,28,28]{2,3,0,1} copy(gte.1)
      bitcast = f32[512,128,28,28]{3,2,1,0} bitcast(transpose)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[512,128,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(copy.1, bitcast, copy.2)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root,
              GmockMatch(m::Tuple(m::Copy(), m::Bitcast(),
                                  m::GetTupleElement(m::Fusion(&fusion)))));
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Tuple(m::Negate(), m::Negate(), m::Copy())));
}

}  // namespace gpu
}  // namespace xla
