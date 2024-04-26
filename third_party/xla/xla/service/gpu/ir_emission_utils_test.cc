/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/gpu/ir_emission_utils.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

class IrEmissionUtilsTest : public HloTestBase {};

TEST_F(IrEmissionUtilsTest, FindTiledLogicalTranspose) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f32[32,48,64]{2,1,0} parameter(0)
  ROOT t = f32[64,32,48]{2,1,0} transpose(p), dimensions={2,0,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* tr = module->entry_computation()->root_instruction();

  auto result = GetDescriptionForTiledTransposeEmitter(*tr, *tr);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, tr);
  EXPECT_EQ(result->dimensions, Vector3({1, 64, 1536}));
  EXPECT_EQ(result->permutation, Vector3({0, 2, 1}));
}

TEST_F(IrEmissionUtilsTest, FindAnyTiledTranspose) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f32[32,48,64]{2,1,0} parameter(0)
  ROOT t = f32[64,48,32]{2,1,0} transpose(p), dimensions={2,1,0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  auto result = GetDescriptionForTiledTransposeEmitter(*r, *r);
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, r);
  EXPECT_EQ(result->dimensions, Vector3({64, 48, 32}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, FindAnyTiledTransposeWithIntermediateUnaryOp) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f32[32,48,64]{2,1,0} parameter(0)
  t = f32[64,48,32]{2,1,0} transpose(p), dimensions={2,1,0}
  ROOT n = f32[64,48,32]{2,1,0} negate(t)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  auto result = GetDescriptionForTiledTransposeEmitter(*r, *r->operand(0));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, r->operand(0));
  EXPECT_EQ(result->dimensions, Vector3({64, 48, 32}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, FindAnyTiledTransposeWithIntermediateUnaryOpS8) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[32,48,64]{2,1,0} parameter(0)
  t = f32[64,48,32]{2,1,0} transpose(p), dimensions={2,1,0}
  ROOT c = s8[64,48,32]{2,1,0} convert(t)
}

ENTRY main {
  p0 = f32[32,48,64]{2,1,0} parameter(0)
  ROOT f = s8[64,48,32]{2,1,0} fusion(p0), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r =
      module->entry_computation()->root_instruction()->fused_expression_root();
  // TODO(b/284431534): Update this test when the shared memory transpose
  // emitter is fast for S8 output.
  EXPECT_FALSE(
      GetDescriptionForTiledTransposeEmitter(*r, *r->operand(0)).has_value());
  EXPECT_EQ(FindNonTrivialHero(*r).name(), "t");
}

TEST_F(IrEmissionUtilsTest, FindReduceHeroEpilogueFusion) {
  const char* hlo = R"(
    HloModule module

    %add {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %fused_computation (param_0.4: f32[128,64], param_1.4: bf16[]) -> bf16[64] {
      %param_0 = f32[128,64]{1,0} parameter(0)
      %param_1 = bf16[] parameter(1)
      %convert.0 = f32[] convert(bf16[] %param_1)
      %reduce.0 = f32[64]{0} reduce(f32[128,64]{1,0} %param_0, f32[] %convert.0), dimensions={0}, to_apply=%add
      ROOT %convert.1 = bf16[64]{0} convert(f32[64]{0} %reduce.0)
    }

    ENTRY %main {
      %param_0 = f32[128,64]{1,0} parameter(0)
      %param_1 = bf16[] parameter(1)
      ROOT fusion = bf16[64]{0} fusion(%param_0, %param_1), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  auto fusion = HloFusionAdaptor::ForInstruction(r);
  const auto& result =
      FindNonTrivialHero(fusion->GetRoots()[0].instruction(), *fusion);
  EXPECT_EQ(result.name(), "reduce.0");
}

TEST_F(IrEmissionUtilsTest, FindReduceHeroEpilogueFusionTwoRootUsers) {
  const char* hlo = R"(
    HloModule module

    Add {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }
    fused_computation {
      param_0 = f32[4,2]{1,0} parameter(0)
      neg = f32[4,2]{1,0} negate(param_0)
      constant_0 = f32[] constant(0)
      reduce.1 = f32[4]{0} reduce(param_0, constant_0), dimensions={1}, to_apply=Add
      bitcast.1 = f32[1,1,4]{2,1,0} bitcast(reduce.1)
      sign.1 = f32[1,1,4]{2,1,0} sign(bitcast.1)
      ROOT tuple.12 = (f32[4,2]{1,0}, f32[1,1,4]{2,1,0}, f32[1,1,4]{2,1,0}) tuple(neg, bitcast.1, sign.1)
    }

    ENTRY main.7749 {
      Arg_2.1 = f32[4,2]{1,0} parameter(0)
      ROOT fusion = (f32[4,2]{1,0}, f32[1,1,4]{2,1,0}, f32[1,1,4]{2,1,0}) fusion(Arg_2.1), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  auto fusion = HloFusionAdaptor::ForInstruction(r);
  const auto& result =
      FindNonTrivialHero(fusion->GetRoots()[1].instruction(), *fusion);
  EXPECT_EQ(result.name(), "reduce.1");
  const auto& result2 =
      FindNonTrivialHero(fusion->GetRoots()[2].instruction(), *fusion);
  EXPECT_EQ(result2.name(), "reduce.1");
}

TEST_F(IrEmissionUtilsTest, FindReduceHeroEpilogueFusionHeroAlsoUsedAsNonHero) {
  const char* hlo = R"(
    HloModule module

    Add {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT add = f32[] add(x, y)
    }

    fused_computation {
      p0 = f32[4]{0} parameter(0)
      zero = f32[] constant(0.0)
      reduce.0 = f32[] reduce(f32[4]{0} p0, f32[] zero), dimensions={0}, to_apply=Add
      broadcast = f32[4]{0} broadcast(f32[] reduce.0), dimensions={}
      reduce.1 = f32[] reduce(f32[4]{0} broadcast, f32[] zero), dimensions={0}, to_apply=Add
      bitcast = f32[1]{0} bitcast(f32[] reduce.0)
      ROOT tuple.1 = (f32[], f32[4]{0}, f32[1]{0}) tuple(f32[] reduce.1, f32[4]{0} broadcast, f32[1]{0} bitcast)
    }

    ENTRY main {
      Arg0 = f32[4]{0} parameter(0)
      ROOT fusion = (f32[], f32[4]{0}, f32[1]{0}) fusion(Arg0), kind=kInput, calls=fused_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  auto fusion = HloFusionAdaptor::ForInstruction(r);
  const auto& result =
      FindNonTrivialHero(fusion->GetRoots()[1].instruction(), *fusion);
  // reduce.0 is also an operand of broadcast, but it is not a hero for that
  // root.
  EXPECT_EQ(result.name(), "broadcast");
  const auto& result2 =
      FindNonTrivialHero(fusion->GetRoots()[2].instruction(), *fusion);
  EXPECT_EQ(result2.name(), "reduce.0");
}

TEST_F(IrEmissionUtilsTest, DoNotFindTransposeHeroEpilogueFusionTwoRootUsers) {
  const char* hlo = R"(
    HloModule module

    fused_computation {
      param_0 = f32[64,32]{1,0} parameter(0)
      transpose = f32[32,64]{1,0} transpose(param_0), dimensions={1,0}
      bitcast.1 = f32[1,32,64]{2,1,0} bitcast(transpose)
      sign.1 = f32[1,32,64]{2,1,0} sign(bitcast.1)
      ROOT tuple.12 = (f32[1,32,64]{2,1,0}, f32[1,32,64]{2,1,0}) tuple(bitcast.1, sign.1)
    }

    ENTRY main.7749 {
      Arg_2.1 = f32[64,32]{1,0} parameter(0)
      ROOT fusion = (f32[1,32,64]{2,1,0}, f32[1,32,64]{2,1,0}) fusion(Arg_2.1), kind=kInput, calls=fused_computation
    }
    )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  auto fusion = HloFusionAdaptor::ForInstruction(r);
  const auto& result =
      FindNonTrivialHero(fusion->GetRoots()[0].instruction(), *fusion);
  EXPECT_EQ(result.name(), "bitcast.1");
  const auto& result2 =
      FindNonTrivialHero(fusion->GetRoots()[1].instruction(), *fusion);
  EXPECT_EQ(result2.name(), "sign.1");
}

TEST_F(IrEmissionUtilsTest, FindAnyTiledTransposeWithIntermediateBinaryOp) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f32[32,48,64]{2,1,0} parameter(0)
  p2 = f32[64,48,32]{2,1,0} parameter(1)
  t = f32[64,48,32]{2,1,0} transpose(p), dimensions={2,1,0}
  ROOT add = f32[64,48,32]{2,1,0} add(t, p2)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();

  auto result = GetDescriptionForTiledTransposeEmitter(*r, *r->operand(0));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, r->operand(0));
  EXPECT_EQ(result->dimensions, Vector3({64, 48, 32}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, FindAnyTiledTransposeWithTwoIntermediateBinaryOps) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[32,48,64]{2,1,0} parameter(0)
  p2 = f32[64,48,32]{2,1,0} parameter(1)
  t = f32[64,48,32]{2,1,0} transpose(p), dimensions={2,1,0}
  mul = f32[64,48,32]{2,1,0} multiply(t, p2)
  ROOT add = f32[64,48,32]{2,1,0} add(mul, p2)
}

ENTRY main {
  param0 = f32[32,48,64]{2,1,0} parameter(0)
  param1 = f32[64,48,32]{2,1,0} parameter(1)
  ROOT fusion = f32[64,48,32]{2,1,0} fusion(param0, param1), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r =
      module->entry_computation()->root_instruction()->fused_expression_root();
  auto result =
      GetDescriptionForTiledTransposeEmitter(*r, FindNonTrivialHero(*r));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, r->operand(0)->operand(0));
  EXPECT_EQ(result->dimensions, Vector3({64, 48, 32}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest,
       FindAnyTiledTransposeWithIntermediateBinaryOpTwoTransposes) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[32,48,64]{2,1,0} parameter(0)
  p2 = f32[48,32,64]{2,1,0} parameter(1)
  t = f32[64,48,32]{2,1,0} transpose(p), dimensions={2,1,0}
  t2 = f32[64,48,32]{2,1,0} transpose(p2), dimensions={2,0,1}
  ROOT add = f32[64,48,32]{2,1,0} add(t, t2)
}

ENTRY main {
  param0 = f32[32,48,64]{2,1,0} parameter(0)
  param1 = f32[48,32,64]{2,1,0} parameter(1)
  ROOT fusion = f32[64,48,32]{2,1,0} fusion(param0, param1), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r =
      module->entry_computation()->root_instruction()->fused_expression_root();
  EXPECT_FALSE(
      GetDescriptionForTiledTransposeEmitter(*r, FindNonTrivialHero(*r))
          .has_value());
  EXPECT_EQ(&FindNonTrivialHero(*r), r);
}

TEST_F(IrEmissionUtilsTest, FindNonTrivialHeroOutsideFusion) {
  const char* hlo = R"(
HloModule module

f {
  p0 = f32[100,200,300]{2,1,0} parameter(0)
  ROOT add = f32[100,200,300]{2,1,0} add(p0, p0)
}

ENTRY entry {
  p0 = f32[300,200,100]{2,1,0} parameter(0)
  t = f32[100,200,300]{2,1,0} transpose(p0), dimensions={2,1,0}
  fusion = f32[100,200,300]{2,1,0} fusion(t), kind=kLoop, calls=f
  ROOT add = f32[100,200,300]{2,1,0} add(t, fusion)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->GetComputationWithName("f")->root_instruction();
  HloInstruction* transpose =
      module->entry_computation()->GetInstructionWithName("t");
  HloInstruction* fusion =
      module->entry_computation()->GetInstructionWithName("fusion");
  auto fusion_adaptor =
      HloFusionAdaptor::ForProducerConsumer(transpose, fusion);
  EXPECT_EQ(&FindNonTrivialHero(*r, *fusion_adaptor), transpose);
}

TEST_F(IrEmissionUtilsTest, FindNonTrivialHeroInsideFusion) {
  const char* hlo = R"(
HloModule module

f {
  p0 = f32[300,200,100]{2,1,0} parameter(0)
  t = f32[100,200,300]{2,1,0} transpose(p0), dimensions={2,1,0}
  ROOT add = f32[100,200,300]{2,1,0} add(t, t)
}

ENTRY entry {
  p0 = f32[300,200,100]{2,1,0} parameter(0)
  p1 = f32[100,200,300]{2,1,0} parameter(1)
  fusion = f32[100,200,300]{2,1,0} fusion(p0), kind=kLoop, calls=f
  ROOT add = f32[100,200,300]{2,1,0} add(p1, fusion)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r = module->entry_computation()->root_instruction();
  HloInstruction* transpose = module->GetComputationWithName("f")
                                  ->parameter_instruction(0)
                                  ->users()
                                  .front();
  HloInstruction* fusion =
      module->entry_computation()->GetInstructionWithName("fusion");
  auto fusion_adaptor = HloFusionAdaptor::ForProducerConsumer(fusion, r);
  EXPECT_EQ(&FindNonTrivialHero(*r, *fusion_adaptor), transpose);
}

TEST_F(IrEmissionUtilsTest, TransposeReachableViaTrivialAndNontrivialOps) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f64[16,16]{1,0} parameter(0)
  trans = f64[16,16]{1,0} transpose(p), dimensions={1,0}
  rev = f64[16,16]{1,0} reverse(trans), dimensions={0,1}
  sub = f64[16,16]{1,0} subtract(trans, trans)
  ROOT add = f64[16,16]{1,0} add(rev, sub)
}

ENTRY main {
  param = f64[16,16]{1,0} parameter(0)
  ROOT fusion = f64[16,16]{1,0} fusion(param), kind=kLoop, calls=fusion
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* r =
      module->entry_computation()->root_instruction()->fused_expression_root();
  EXPECT_FALSE(
      GetDescriptionForTiledTransposeEmitter(*r, FindNonTrivialHero(*r))
          .has_value());
  EXPECT_EQ(&FindNonTrivialHero(*r), r);
}

TEST_F(IrEmissionUtilsTest, FindTiledTransposeOneSwapDimIsSmall) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[100,11,12,8]{3,2,1,0} parameter(0)
  ROOT c = f32[100,11,12,8]{1,0,2,3} copy(p)
}

ENTRY main {
  param = f32[100,11,12,8]{3,2,1,0} parameter(0)
  ROOT fusion = f32[100,11,12,8]{1,0,2,3} fusion(param), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* copy =
      module->entry_computation()->root_instruction()->fused_expression_root();
  auto result =
      GetDescriptionForTiledTransposeEmitter(*copy, FindNonTrivialHero(*copy));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, copy);
  EXPECT_EQ(result->dimensions, Vector3({8, 12, 1100}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, FindTiledLogicalTransposeOneSwapDimIsSmall) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[100,11,12,8]{3,2,1,0} parameter(0)
  ROOT t = f32[8,12,100,11]{3,2,1,0} transpose(p), dimensions={3,2,0,1}
}

ENTRY main {
  param = f32[100,11,12,8]{3,2,1,0} parameter(0)
  ROOT fusion = f32[8,12,100,11]{3,2,1,0} fusion(param), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* tr =
      module->entry_computation()->root_instruction()->fused_expression_root();
  auto result =
      GetDescriptionForTiledTransposeEmitter(*tr, FindNonTrivialHero(*tr));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, tr);
  EXPECT_EQ(result->dimensions, Vector3({8, 12, 1100}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, FindTiledTransposeOtherSwapDimIsSmall) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[8,12,100,11]{3,2,1,0} parameter(0)
  ROOT c = f32[8,12,100,11]{0,1,3,2} copy(p)
}

ENTRY main {
  param = f32[8,12,100,11]{3,2,1,0} parameter(0)
  ROOT fusion = f32[8,12,100,11]{0,1,3,2} fusion(param), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* copy =
      module->entry_computation()->root_instruction()->fused_expression_root();
  auto result =
      GetDescriptionForTiledTransposeEmitter(*copy, FindNonTrivialHero(*copy));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, copy);
  EXPECT_EQ(result->dimensions, Vector3({1100, 12, 8}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, FindTiledLogicalTransposeOtherSwapDimIsSmall) {
  const char* hlo = R"(
HloModule module

fusion {
  p = f32[8,12,100,11]{3,2,1,0} parameter(0)
  ROOT t = f32[100,11,12,8]{3,2,1,0} transpose(p), dimensions={2,3,1,0}
}

ENTRY main {
  param = f32[8,12,100,11]{3,2,1,0} parameter(0)
  ROOT fusion = f32[100,11,12,8]{3,2,1,0} fusion(param), kind=kInput, calls=fusion
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* tr =
      module->entry_computation()->root_instruction()->fused_expression_root();
  auto result =
      GetDescriptionForTiledTransposeEmitter(*tr, FindNonTrivialHero(*tr));
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result->instr, tr);
  EXPECT_EQ(result->dimensions, Vector3({1100, 12, 8}));
  EXPECT_EQ(result->permutation, Vector3({2, 1, 0}));
}

TEST_F(IrEmissionUtilsTest, IsContiguousSlice) {
  const char* hlo = R"(
HloModule module

ENTRY entry {
  p = f32[8,12,100,11]{3,2,1,0} parameter(0)
  slice.1 = f32[2,12,100,11]{3,2,1,0} slice(p), slice={[1:3], [0:12], [0:100], [0:11]}
  slice.2 = f32[1,1,1,11]{3,2,1,0} slice(p), slice={[1:2], [0:1], [0:1], [0:11]}
  slice.3 = f32[1,1,10,11]{3,2,1,0} slice(p), slice={[1:2], [0:1], [0:10], [0:11]}
  slice.4 = f32[1,2,10,11]{3,2,1,0} slice(p), slice={[1:2], [0:2], [0:10], [0:11]}
  slice.5 = f32[8,2,100,11]{3,2,1,0} slice(p), slice={[0:8], [10:12], [0:100], [0:11]}
  c = f32[8,12,100,11]{0,1,3,2} copy(p)
  slice.6 = f32[8,12,40,11]{0,1,3,2} slice(c), slice={[0:8], [0:12], [10:50], [0:11]}
  slice.7 = f32[8,12,1,2]{0,1,3,2} slice(c), slice={[0:8], [0:12], [0:1], [0:2]}
  slice.8 = f32[8,2,100,11]{0,1,3,2} slice(c), slice={[0:8], [0:2], [0:100], [0:11]}
  slice.9 = f32[8,2,40,11]{0,1,3,2} slice(c), slice={[0:8], [10:12], [10:50], [0:11]}
  slice.10 = f32[8,2,50,11]{3,2,1,0} slice(p), slice={[0:8:1], [10:12:1], [0:100:2], [0:11:1]}
  ROOT t = (f32[2,12,100,11]{3,2,1,0},
            f32[1,1,1,11]{3,2,1,0},
            f32[1,1,10,11]{3,2,1,0},
            f32[1,2,10,11]{3,2,1,0},
            f32[8,2,100,11]{3,2,1,0},
            f32[8,12,40,11]{0,1,3,2},
            f32[8,12,1,2]{0,1,3,2},
            f32[8,2,100,11]{0,1,3,2},
            f32[8,2,40,11]{0,1,3,2},
            f32[8,2,50,11]{3,2,1,0}) tuple(slice.1, slice.2, slice.3, slice.4, slice.5, slice.6, slice.7, slice.8, slice.9, slice.10)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));

  HloInstruction* slice1 =
      module->entry_computation()->GetInstructionWithName("slice.1");
  HloInstruction* slice2 =
      module->entry_computation()->GetInstructionWithName("slice.2");
  HloInstruction* slice3 =
      module->entry_computation()->GetInstructionWithName("slice.3");
  HloInstruction* slice4 =
      module->entry_computation()->GetInstructionWithName("slice.4");
  HloInstruction* slice5 =
      module->entry_computation()->GetInstructionWithName("slice.5");
  HloInstruction* slice6 =
      module->entry_computation()->GetInstructionWithName("slice.6");
  HloInstruction* slice7 =
      module->entry_computation()->GetInstructionWithName("slice.7");
  HloInstruction* slice8 =
      module->entry_computation()->GetInstructionWithName("slice.8");
  HloInstruction* slice9 =
      module->entry_computation()->GetInstructionWithName("slice.9");
  HloInstruction* slice10 =
      module->entry_computation()->GetInstructionWithName("slice.10");
  EXPECT_TRUE(IsContiguousSlice(*slice1));
  EXPECT_TRUE(IsContiguousSlice(*slice2));
  EXPECT_TRUE(IsContiguousSlice(*slice3));
  EXPECT_TRUE(!IsContiguousSlice(*slice4));
  EXPECT_TRUE(!IsContiguousSlice(*slice5));
  EXPECT_TRUE(IsContiguousSlice(*slice6));
  EXPECT_TRUE(IsContiguousSlice(*slice7));
  EXPECT_TRUE(!IsContiguousSlice(*slice8));
  EXPECT_TRUE(!IsContiguousSlice(*slice9));
  EXPECT_TRUE(!IsContiguousSlice(*slice10));
}

TEST_F(IrEmissionUtilsTest, LiteralToAttrToXlaFormat) {
  // int16, should be aliased.
  {
    Literal literal = LiteralUtil::CreateR2<int16_t>({{0, 1, 2}, {3, 4, 5}});

    TF_ASSERT_OK_AND_ASSIGN(DenseDataIntermediate data,
                            LiteralToXlaFormat(literal));
    EXPECT_EQ(data.span().size(), literal.size_bytes());
    EXPECT_EQ(reinterpret_cast<const char*>(data.span().data()),
              literal.untyped_data());
  }

  // int4, even, should be a new (unaliased) packed array.
  {
    Literal literal = LiteralUtil::CreateR2<s4>(
        {{s4(0), s4(1), s4(2)}, {s4(3), s4(4), s4(5)}});

    TF_ASSERT_OK_AND_ASSIGN(DenseDataIntermediate data,
                            LiteralToXlaFormat(literal));
    EXPECT_EQ(data.span(), std::vector<uint8_t>({0x01, 0x23, 0x45}));
    EXPECT_NE(reinterpret_cast<const void*>(data.span().data()),
              literal.untyped_data());
  }

  // int4, odd, should be a new (unaliased) packed array.
  {
    Literal literal = LiteralUtil::CreateR2<u4>(
        {{u4(0), u4(1), u4(2)}, {u4(3), u4(4), u4(5)}, {u4(6), u4(7), u4(8)}});

    TF_ASSERT_OK_AND_ASSIGN(DenseDataIntermediate data,
                            LiteralToXlaFormat(literal));
    EXPECT_EQ(data.span(),
              std::vector<uint8_t>({0x01, 0x23, 0x45, 0x67, 0x80}));
    EXPECT_NE(reinterpret_cast<const void*>(data.span().data()),
              literal.untyped_data());
  }
}

}  // namespace gpu
}  // namespace xla
