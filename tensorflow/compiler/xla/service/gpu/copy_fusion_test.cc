/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/copy_fusion.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

namespace op = xla::testing::opcode_matchers;

class CopyFusionTest : public HloTestBase {
 public:
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
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement()));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Transpose(), op::Copy(), op::Copy()));
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
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement()));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Negate(), op::Copy(), op::Copy()));
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

TEST_F(CopyFusionTest, CopyFusionShouldNotRunWithDynamicUpdateSlice) {
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

TEST_F(CopyFusionTest, CopyFusionTransposeAndThreeCopies) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      param_0.1 = f32[16,32]{1,0} parameter(0)
      s.1 = f32[16,32]{1,0} sqrt(param_0.1)
      ROOT c.1 = f32[32,16]{1,0} transpose(s.1), dimensions={1,0}
    }

    ENTRY entry {
      p = f32[16,32]{1,0} parameter(0)
      fusion = f32[32,16]{1,0} fusion(p), kind=kInput, calls=fused_computation
      copy.1 = f32[32,16]{1,0} copy(fusion)
      copy.2 = f32[32,16]{1,0} copy(fusion)
      copy.3 = f32[32,16]{1,0} copy(fusion)
      ROOT root = (f32[32,16]{1,0}, f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(copy.1, copy.2, copy.3)
    })"))
                    .value();
  ASSERT_TRUE(cf_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement()));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Transpose(), op::Copy(), op::Copy(), op::Copy()));
}

TEST_F(CopyFusionTest, CopyFusionShouldNotRunWithOnlyOneCopie) {
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
      ROOT root = (f32[128,512,28,28]{3,2,1,0}) tuple(copy.1)
    })"))
                    .value();
  ASSERT_FALSE(cf_.Run(module.get()).value());
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
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::Bitcast(),
                              op::GetTupleElement()));
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Negate(), op::Copy(), op::Copy()));
}

TEST_F(CopyFusionTest, CopyFusionShouldNotRunWithOnlyOneNonTransposeCopie) {
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
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

TEST_F(CopyFusionTest, CopyFusionShouldConcatenateCopiesWithTuples) {
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
  // TODO (b/285561974): Make Copy Fusion pass works with multi-output fusion.
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
      copy.1 = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) copy(fusion)
      copy.2 = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) copy(fusion)
      gte0 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(copy.1), index=0
      gte1 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(copy.2), index=1
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(gte0, gte1)
    })"))
                    .value();
  // TODO (b/285561974): Make Copy Fusion pass works with multi-output fusion.
  ASSERT_FALSE(cf_.Run(module.get()).value());
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
      copy.1 = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) copy(fusion)
      copy.2 = (f32[128,512,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) copy(fusion)
      gte0 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(copy.1), index=0
      gte1 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(copy.2), index=1
      gte2 = f32[128,512,28,28]{3,2,1,0} get-tuple-element(fusion), index=0
      transpose = f32[128,512,28,28]{2,3,0,1} copy(gte2)
      bitcast = f32[512,128,28,28]{3,2,1,0} bitcast(transpose)
      ROOT root = (f32[128,512,28,28]{3,2,1,0}, f32[512,128,28,28]{3,2,1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(gte0, bitcast, gte1)
    })"))
                    .value();
  // TODO (b/285561974): Make Copy Fusion pass works with multi-output fusion.
  ASSERT_FALSE(cf_.Run(module.get()).value());
}

}  // namespace gpu
}  // namespace xla
