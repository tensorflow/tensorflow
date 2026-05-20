/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/analysis/alias_info.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

class GetInPlaceInputOutputPairsTest : public HloHardwareIndependentTestBase {
 protected:
  AliasInfo alias_info_;
};

// Verifies that by default, a call-start/async-start operation does not alias
// its operands with its outputs.
TEST_F(GetInPlaceInputOutputPairsTest, AsyncStartDefaultAliasing) {
  const char* const kHlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0),
    to_apply=async_computation
  ROOT done = f32[2,3] call-done(start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* start = FindInstruction(module.get(), "start");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(start);
  EXPECT_THAT(pairs, IsEmpty());
}

// Verifies that by default, a call-start/async-start operation aliases its
// operand with the corresponding output defined in output_to_operand_aliasing.
TEST_F(GetInPlaceInputOutputPairsTest, AsyncStartForwardedOperand) {
  const char* const kHlo = R"(
HloModule test

async_computation {
  p0 = f32[2,3] parameter(0)
  ROOT abs = f32[2,3] abs(p0)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  start = ((f32[2,3]), f32[2,3], s32[]) call-start(p0),
    to_apply=async_computation,
    output_to_operand_aliasing={{1}: (0, {})}
  ROOT done = f32[2,3] call-done(start)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* start = FindInstruction(module.get(), "start");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(start);

  // By default for forwarded operands: operand 0 maps to output {0, 0} for the
  // parameter subshape
  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{0, {}}, {1}}));
}

// Verifies that a dynamic-update-slice instruction can compute in-place
// with its first operand (the array being updated).
TEST_F(GetInPlaceInputOutputPairsTest, DUS) {
  const char* kHlo = R"(
HloModule test

ENTRY test {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT dus = f32[10] dynamic-update-slice(p0, p1, p2)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* dus = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(dus);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that a fusion containing a dynamic-update-slice as root can
// compute in-place with the corresponding parameter.
TEST_F(GetInPlaceInputOutputPairsTest, DUSFusion) {
  const char* kHlo = R"(
HloModule test

fused_computation {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT dus = f32[10] dynamic-update-slice(p0, p1, p2)
}

ENTRY test {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = f32[10] fusion(p0, p1, p2), kind=kLoop,
      calls=fused_computation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that GetInPlaceInputOutputPairs respects output_to_operand_aliasing
// annotations on async-start instructions.
TEST_F(GetInPlaceInputOutputPairsTest, AsyncStartWithOutputOperandAliasing) {
  const char* kHlo = R"(
HloModule module

%async_computation {
  %param_0.2 = (f32[8,4,1],
      (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)})) parameter(0)
  %get-tuple-element = f32[8,4,1] get-tuple-element(%param_0.2), index=0
  ROOT %all-to-all0.0 = f32[8,4,1] all-to-all(%get-tuple-element),
      channel_id=1, replica_groups={{0,1,2,3,4,5,6,7}},
      dimensions={0}
}

ENTRY %Comp_spmd {
  %param = f32[8,4,1] parameter(0)
  %copy = f32[8,4,1] copy(%param)
  %custom-call = (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)}) custom-call(),
      custom_call_target="BarrierStart"
  %tuple = (f32[8,4,1], (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)}))
      tuple(%copy, %custom-call)
  %all-to-all-start.1 = (((f32[8,4,1],
      (f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)}))),
      f32[8,4,1], u32[]{:S(2)}, u32[]{:S(2)})
      async-start(%tuple),
      output_to_operand_aliasing={
        {0,0,1,0}: (0, {1,0}),
        {0,0,1,1}: (0, {1,1}),
        {0,0,1,2}: (0, {1,2})
      },
      calls=%async_computation
  %all-to-all-done = f32[8,4,1] async-done(%all-to-all-start.1)
  ROOT %copy.1 = f32[8,4,1] copy(%all-to-all-done)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* async_start = module->entry_computation()
                                    ->root_instruction()
                                    ->mutable_operand(0)
                                    ->mutable_operand(0);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(async_start);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back(
      {HloOperandIndex{0, {1, 0}}, {0, 0, 1, 0}});  // annotated
  expected_pairs.push_back(
      {HloOperandIndex{0, {1, 1}}, {0, 0, 1, 1}});  // annotated
  expected_pairs.push_back(
      {HloOperandIndex{0, {1, 2}}, {0, 0, 1, 2}});  // annotated
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that a fusion with output_to_operand_aliasing annotation returns
// both discovered in-place pairs (from DUS) and annotated ones.
TEST_F(GetInPlaceInputOutputPairsTest, DUSFusionWithOutputOperandAliasing) {
  const char* kHlo = R"(
HloModule test

fused_computation {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  dus = f32[10] dynamic-update-slice(p0, p1, p2)
  ROOT tuple = (f32[5], f32[10]) tuple(p1, dus)
}

ENTRY test {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = (f32[5], f32[10]) fusion(p0, p1, p2), kind=kLoop,
      output_to_operand_aliasing={{0}: (1, {})}, calls=fused_computation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {1}});  // discovered
  expected_pairs.push_back({HloOperandIndex{1, {}}, {0}});  // annotated
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that a fusion that does not contain DUS and has no aliasing
// annotations returns empty in-place pairs.
TEST_F(GetInPlaceInputOutputPairsTest, NonDUSFusion) {
  const char* kHlo = R"(
HloModule test

fused_computation {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT add = f32[10] add(p0, p1)
}

ENTRY test {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT fusion = f32[10] fusion(p0, p1), kind=kLoop, calls=fused_computation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  EXPECT_THAT(in_place_pairs, IsEmpty());
}

// Verifies that a fusion with no DUS but with aliasing annotations returns
// the annotated pairs.
TEST_F(GetInPlaceInputOutputPairsTest, NonDUSFusionWithOutputOperandAliasing) {
  const char* kHlo = R"(
HloModule test

fused_computation {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT add = f32[10] add(p0, p1)
}

ENTRY test {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT fusion = f32[10] fusion(p0, p1), kind=kLoop,
      output_to_operand_aliasing={{}: (0, {})}, calls=fused_computation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);

  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that in-place computation works through nested fusions when the
// inner fusion contains a dynamic-update-slice.
TEST_F(GetInPlaceInputOutputPairsTest, NestedDUSFusion) {
  const char* kHlo = R"(
HloModule test

fused_computation1 {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT dus = f32[10] dynamic-update-slice(p0, p1, p2)
}

fused_computation2 {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = f32[10] fusion(p0, p1, p2), kind=kLoop,
      calls=fused_computation1
}

ENTRY test {
  p0 = f32[10] parameter(0)
  p1 = f32[5] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = f32[10] fusion(p0, p1, p2), kind=kLoop,
      calls=fused_computation2
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies in-place behavior for nested multi-output fusions containing DUS.
TEST_F(GetInPlaceInputOutputPairsTest, NestedMultiOutputDUSFusion) {
  const char* kHlo = R"(
HloModule test

fused_computation1 {
  p0 = s32[] parameter(0)
  p1 = (f32[5],f32[10]) parameter(1)
  gte0 = f32[5] get-tuple-element(p1), index=0
  gte1 = f32[10] get-tuple-element(p1), index=1
  dus = f32[10] dynamic-update-slice(gte1, gte0, p0)
  negate = f32[5] negate(gte0)
  ROOT tuple = (f32[5],f32[10]) tuple(negate, dus)
}

fused_computation2 {
  p0 = f32[5] parameter(0)
  p1 = (f32[10],s32[]) parameter(1)
  gte0 = f32[10] get-tuple-element(p1), index=0
  gte1 = s32[] get-tuple-element(p1), index=1
  in_tuple = (f32[5],f32[10]) tuple(p0, gte0)
  inner_fusion = (f32[5],f32[10]) fusion(gte1, in_tuple), kind=kLoop,
      calls=fused_computation1
  fusion_gte0 = f32[5] get-tuple-element(inner_fusion), index=0
  fusion_gte1 = f32[10] get-tuple-element(inner_fusion), index=1
  negate = f32[5] negate(p0)
  ROOT tuple = (f32[5],f32[5],f32[10])
      tuple(negate, fusion_gte0, fusion_gte1)
}

ENTRY test {
  p0 = f32[5] parameter(0)
  p1 = (f32[10],s32[]) parameter(1)
  ROOT fusion = (f32[5],f32[5],f32[10]) fusion(p0, p1), kind=kLoop,
      calls=fused_computation2
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  HloInstruction* inner_fusion = FindInstruction(module.get(), "inner_fusion");

  auto inner_in_place_pairs =
      alias_info_.GetInPlaceInputOutputPairs(inner_fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> inner_expected_pairs;
  inner_expected_pairs.push_back({HloOperandIndex{1, {1}}, {1}});
  EXPECT_EQ(inner_in_place_pairs, inner_expected_pairs);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{1, {0}}, {2}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies in-place behavior when there is aliasing within a nested loop
// fusion.
TEST_F(GetInPlaceInputOutputPairsTest, NestedLoopWithAliasingInDUSFusion) {
  const char* kHlo = R"(
HloModule test

copy_fusion {
  input = s8[8,256,1,256] parameter(0)
  ROOT copy.3 = s8[8,256,1,256] copy(input)
}

fused_computation.0 {
  p0 = (s8[8,256,1,256],s8[1,256,1,256]) parameter(0)
  gte0 = s8[8,256,1,256] get-tuple-element(p0), index=0
  gte1 = s8[1,256,1,256] get-tuple-element(p0), index=1
  fusion = s8[8,256,1,256] fusion(gte0), kind=kLoop,
      output_to_operand_aliasing={{}: (0, {})}, calls=copy_fusion
  p1 = s8[1,256,1,256] parameter(1)
  added = s8[1,256,1,256] add(gte1, p1)
  p2 = s32[] parameter(2)
  c0 = s32[] constant(0)
  ROOT dynamic-update-slice.0 = s8[8,256,1,256]
      dynamic-update-slice(fusion, added, p2, c0, c0, c0)
}

ENTRY test {
  p0 = (s8[8,256,1,256],s8[1,256,1,256]) parameter(0)
  p1 = s8[1,256,1,256] parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion = s8[8,256,1,256] fusion(p0, p1, p2), kind=kLoop,
      calls=fused_computation.0
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {0}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies in-place behavior for a DUS fusion that also involves collective
// operations like all-reduce.
TEST_F(GetInPlaceInputOutputPairsTest, DUSLoopFusionWithCollective) {
  const char* kHlo = R"(
HloModule LoopFusionAllReduce

fused_computation.1 {
  p0 = bf16[2,8192,6144]{2,1,0:T(8,128)(2,1)} parameter(0)
  ROOT slice = bf16[2,2048,6144]{2,1,0:T(8,128)(2,1)} slice(p0),
      slice={[0:2], [6144:8192], [0:6144]}
}

fused_computation.2 {
  p0 = bf16[2,8192]{1,0:T(2,128)(2,1)} parameter(0)
  ROOT slice = bf16[2,2048]{1,0:T(2,128)(2,1)} slice(p0),
      slice={[0:2], [6144:8192]}
}

sum {
  lhs = bf16[] parameter(0)
  rhs = bf16[] parameter(1)
  ROOT add = bf16[] add(lhs, rhs)
}

fused_computation {
  p0 = bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)} parameter(0)
  p1 = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} parameter(1)
  p2 = bf16[2,8192,6144]{2,1,0:T(8,128)(2,1)} parameter(2)
  p3 = bf16[2,8192]{1,0:T(2,128)(2,1)} parameter(3)
  fusion.1 = bf16[2,2048,6144]{2,1,0:T(8,128)(2,1)} fusion(p2),
      kind=kLoop, calls=fused_computation.1
  bitcast = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} bitcast(fusion.1)
  fusion.2 = bf16[2,2048]{1,0:T(2,128)(2,1)} fusion(p3), kind=kLoop,
      calls=fused_computation.2
  broadcast = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)}
      broadcast(fusion.2),
      dimensions={1,2}
  multiply = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)S(1)}
      multiply(bitcast, broadcast)
  all-reduce = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} all-reduce(p1),
      replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=sum
  c0 = u32[] constant(0)
  c1 = u32[] constant(4096)
  dynamic-update-slice = bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)}
      dynamic-update-slice(p0, all-reduce, c0, c0, c1, c0)
  ROOT tuple = (bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)S(1)},
      bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)})
      tuple(multiply, dynamic-update-slice)
}

ENTRY entry {
  p0 = bf16[2,8192,6144]{2,1,0:T(8,128)(2,1)} parameter(0)
  p1 = bf16[2,8192]{1,0:T(2,128)(2,1)} parameter(1)
  p2 = bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)} parameter(2)
  p3 = bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)} parameter(3)
  ROOT fusion = (bf16[1,2,2048,6144]{3,2,1,0:T(8,128)(2,1)S(1)},
      bf16[1,2,8192,6144]{3,2,1,0:T(8,128)(2,1)})
      fusion(p3, p2, p0, p1), kind=kLoop, calls=fused_computation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {1}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies in-place behavior for a DUS output fusion involving collectives.
TEST_F(GetInPlaceInputOutputPairsTest, DUSOutputFusionWithCollective) {
  const char* kHlo = R"(
HloModule OutputFusionAllReduce

fused_computation.0 {
  p0 = bf16[4096,9216]{1,0:T(8,128)(2,1)} parameter(0)
  ROOT slice = bf16[1024,9216]{1,0:T(8,128)(2,1)} slice(p0),
      slice={[3072:4096], [0:9216]}
}

fused_computation.1 {
  p0 = s8[9216,6144]{1,0:T(8,128)(4,1)S(1)} parameter(0)
  ROOT bitcast = s8[9216,6144]{1,0:T(8,128)(4,1)} bitcast(p0)
}

add {
  x = bf16[] parameter(0)
  y = bf16[] parameter(1)
  ROOT add = bf16[] add(x, y)
}

fused_computation {
  p0 = bf16[4096,6144]{1,0:T(8,128)(2,1)} parameter(0)
  p1 = bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)} parameter(1)
  p2 = bf16[4096,9216]{1,0:T(8,128)(2,1)} parameter(2)
  p3 = s8[9216,6144]{1,0:T(8,128)(4,1)S(1)} parameter(3)
  fusion1 = bf16[1024,9216]{1,0:T(8,128)(2,1)} fusion(p2), kind=kLoop,
      calls=fused_computation.0
  fusion2 = s8[9216,6144]{1,0:T(8,128)(4,1)} fusion(p3), kind=kLoop,
      calls=fused_computation.1
  convolution = bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)}
      convolution(fusion1, fusion2), dim_labels=bf_io->bf
  all-reduce = bf16[1024,6144]{1,0:T(8,128)(2,1)} all-reduce(p1),
      replica_groups={{0,1,2,3,4,5,6,7}}, to_apply=add
  c1 = u32[] constant(2048)
  c0 = u32[] constant(0)
  dynamic-update-slice = bf16[4096,6144]{1,0:T(8,128)(2,1)}
      dynamic-update-slice(p0, all-reduce, c1, c0)
  ROOT tuple = (bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)},
      bf16[4096,6144]{1,0:T(8,128)(2,1)})
      tuple(convolution, dynamic-update-slice)
}

ENTRY entry {
  p0 = bf16[4096,9216]{1,0:T(8,128)(2,1)} parameter(0)
  p1 = s8[9216,6144]{1,0:T(8,128)(4,1)S(1)} parameter(1)
  p2 = bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)} parameter(2)
  p3 = bf16[4096,6144]{1,0:T(8,128)(2,1)} parameter(3)
  ROOT fusion = (bf16[1024,6144]{1,0:T(8,128)(2,1)S(1)},
      bf16[4096,6144]{1,0:T(8,128)(2,1)})
      fusion(p3, p2, p0, p1), kind=kOutput, calls=fused_computation
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{0, {}}, {1}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that DUS can share buffer through a bitcast in a fusion.
TEST_F(GetInPlaceInputOutputPairsTest, DUSLoopFusionWithBitcast) {
  const char* kHlo = R"(
HloModule DUSLoopFusionWithBitcast

fused_dynamic_update_slice {
  param_1.133 = bf16[32,1,4096,18432]{2,3,1,0} parameter(1)
  bitcast.8539.1 = bf16[32,1,18432,4096]{3,2,1,0} bitcast(param_1.133)
  param_0.168 = bf16[1,4096,18432]{1,0,2} parameter(0)
  bitcast.8543.1 = bf16[1,1,18432,4096]{3,2,1,0} bitcast(param_0.168)
  param_2.98 = s32[] parameter(2)
  constant_2153_8 = s32[] constant(0)
  compare.753.6 = pred[] compare(param_2.98, constant_2153_8), direction=LT
  constant_2154_12 = s32[] constant(96)
  add.950.6 = s32[] add(param_2.98, constant_2154_12)
  select.883.5 = s32[] select(compare.753.6, add.950.6, param_2.98)
  ROOT dynamic-update-slice.178.1 = bf16[32,1,18432,4096]{3,2,1,0}
      dynamic-update-slice(bitcast.8539.1, bitcast.8543.1, select.883.5,
          constant_2153_8, constant_2153_8, /*index=5*/constant_2153_8)
}

ENTRY entry {
  p0 = bf16[1,4096,18432]{1,0,2} parameter(0)
  p1 = bf16[32,1,4096,18432]{2,3,1,0} parameter(1)
  p2 = s32[] parameter(2)
  ROOT fusion1 = bf16[32,1,18432,4096]{3,2,1,0} fusion(p0, p1, p2),
      kind=kLoop, calls=fused_dynamic_update_slice
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(fusion);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  // p1 should be aliased with fusion1
  expected_pairs.push_back({HloOperandIndex{1, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies in-place behavior for ragged-all-to-all instruction.
TEST_F(GetInPlaceInputOutputPairsTest, RaggedAllToAll) {
  const char* kHlo = R"(
HloModule RaggedAllToAll, is_scheduled=true

ENTRY AllToAll {
  input = f32[24,56,119] parameter(0)
  copy-start = (f32[24,56,119], f32[24,56,119], u32[]) copy-start(input)
  c0 = f32[] constant(0)
  output = f32[24,56,119] broadcast(c0), dimensions={}
  input_offsets = s32[8] parameter(1)
  send_sizes = s32[8] parameter(2)
  output_offsets = s32[8] parameter(3)
  recv_sizes = s32[8] parameter(4)
  copy-done = f32[24,56,119] copy-done(copy-start)
  ROOT ra2a = f32[24,56,119] ragged-all-to-all(copy-done, output,
      input_offsets, send_sizes, output_offsets, recv_sizes),
      replica_groups={{0,1,2,3,4,5,6,7}}
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloInstruction* ragged_all_to_all =
      module->entry_computation()->root_instruction();

  auto in_place_pairs =
      alias_info_.GetInPlaceInputOutputPairs(ragged_all_to_all);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  expected_pairs.push_back({HloOperandIndex{1, {}}, {}});
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that all-reduce-start with NVSHMEM backend expects no aliasing
// (empty in-place pairs).
TEST_F(GetInPlaceInputOutputPairsTest, nvshmem_ar) {
  const char* kHlo = R"(
HloModule test_ar
region_add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT ret = f32[] add(lhs, rhs)
}

ENTRY test {
  p0 = f32[10] parameter(0)
  ar = f32[10] all-reduce-start(p0), replica_groups={},
      to_apply=region_add,
      backend_config={
        "collective_backend_config":
          {
            "backend":"NVSHMEM"
          }
        }
  ROOT ar.done = f32[10] all-reduce-done(ar)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* ar_start =
      module->entry_computation()->root_instruction()->operand(0);

  auto in_place_pairs = alias_info_.GetInPlaceInputOutputPairs(ar_start);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  // For nvshmem allreduce, we expect no aliasing for input and output buffers
  // therefore empty inplace pairs.
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that collective-permute-start expects no aliasing (empty in-place
// pairs).
TEST_F(GetInPlaceInputOutputPairsTest, CombinedCollectivePermute) {
  const char* kHlo = R"(
HloModule test_cp
ENTRY test {
  p0 = f32[2,128]{1,0} parameter(0)
  p1 = f32[2,128]{1,0} parameter(1)
  p2 = f32[2,128]{1,0} parameter(2)
  p3 = f32[2,128]{1,0} parameter(3)
  collective-permute-start.0 = ((f32[2,128]{1,0}, f32[2,128]{1,0},
      f32[2,128]{1,0}, f32[2,128]{1,0}), (f32[2,128]{1,0},
      f32[2,128]{1,0}, f32[2,128]{1,0}, f32[2,128]{1,0}))
      collective-permute-start(p0, p1, p2, p3), channel_id=0,
      source_target_pairs={{0,2},{2,4},{1,3},{3,5}},
      backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "collective_backend_config":{
              "is_sync":false,
              "is_pipelined":false,
              "backend":"DEFAULT"
          },
          "force_earliest_schedule":false,"reification_cost":[],
          "device_type":"DEVICE_TYPE_INVALID"}
  ROOT collective-permute-done.0 = (f32[2,128]{1,0}, f32[2,128]{1,0},
      f32[2,128]{1,0}, f32[2,128]{1,0})
      collective-permute-done(collective-permute-start.0)
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* collective_permute_start =
      module->entry_computation()->root_instruction()->operand(0);

  auto in_place_pairs =
      alias_info_.GetInPlaceInputOutputPairs(collective_permute_start);
  std::vector<std::pair<HloOperandIndex, ShapeIndex>> expected_pairs;
  // We expect no aliasing for input and output buffers
  // therefore empty inplace pairs.
  EXPECT_EQ(in_place_pairs, expected_pairs);
}

// Verifies that all-reduce-start with default backend expects aliasing.
TEST_F(GetInPlaceInputOutputPairsTest, AllReduceStartDefault) {
  const char* const kHlo = R"(
HloModule test

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  ars = f32[2,3] all-reduce-start(p0), replica_groups={}, to_apply=add
  ROOT ard = f32[2,3] all-reduce-done(ars)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* ars = FindInstruction(module.get(), "ars");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(ars);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{0, {}}, {}}));
}

// Verifies in-place behavior for multi-operand all-reduce-start.
TEST_F(GetInPlaceInputOutputPairsTest, AllReduceStartMultiOperand) {
  const char* const kHlo = R"(
HloModule test

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  ars = (f32[2,3], f32[2,3]) all-reduce-start(p0, p1),
      replica_groups={}, to_apply=add
  ROOT ard = (f32[2,3], f32[2,3]) all-reduce-done(ars)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* ars = FindInstruction(module.get(), "ars");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(ars);

  EXPECT_THAT(
      pairs,
      ElementsAre(
          std::pair<HloOperandIndex, ShapeIndex>{HloOperandIndex{0, {}}, {0}},
          std::pair<HloOperandIndex, ShapeIndex>{HloOperandIndex{1, {}}, {1}}));
}

TEST_F(GetInPlaceInputOutputPairsTest, ScatterDefaultAliasing) {
  const char* const kHlo = R"(
HloModule test

update_computation {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

ENTRY main {
  p0 = f32[10] parameter(0)
  p1 = s32[] parameter(1)
  p2 = f32[] parameter(2)
  ROOT scatter = f32[10] scatter(p0, p1, p2),
      update_window_dims={},
      inserted_window_dims={0},
      scatter_dims_to_operand_dims={0},
      index_vector_dim=0,
      to_apply=update_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* scatter = FindInstruction(module.get(), "scatter");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(scatter);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{0, {}}, {}}));
}

TEST_F(GetInPlaceInputOutputPairsTest, CollectivePermuteStartInplaceTuple) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = (f32[128,32], f32[128,32]) parameter(0)
  p1 = (f32[128,128], f32[128,128]) parameter(1)
  p2 = ((s32[], s32[]), (s32[], s32[])) parameter(2)
  p3 = ((s32[], s32[]), (s32[], s32[])) parameter(3)
  cps = ((f32[128,32], f32[128,32]), (f32[128,128], f32[128,128]))
      collective-permute-start(p0, p1, p2, p3), source_target_pairs={{0,1}},
      slice_sizes={{128,32}, {128,32}}
  ROOT cpd = (f32[128,128], f32[128,128])
      collective-permute-done(cps)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* cps = FindInstruction(module.get(), "cps");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(cps);

  EXPECT_THAT(pairs, ElementsAre(
                         std::pair<HloOperandIndex, ShapeIndex>{
                             HloOperandIndex{1, {}}, {1}},
                         std::pair<HloOperandIndex, ShapeIndex>{
                             HloOperandIndex{1, {0}}, {1, 0}},
                         std::pair<HloOperandIndex, ShapeIndex>{
                             HloOperandIndex{1, {1}}, {1, 1}}));
}

// Verifies in-place behavior for collective-permute-start with non-tuple
// output.
TEST_F(GetInPlaceInputOutputPairsTest, CollectivePermuteStartInplaceNonTuple) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[128,32] parameter(0)
  p1 = f32[128,128] parameter(1)
  p2 = (s32[], s32[]) parameter(2)
  p3 = (s32[], s32[]) parameter(3)
  cps = (f32[128,32], f32[128,128]) collective-permute-start(p0, p1, p2, p3),
      source_target_pairs={{0,1}}, slice_sizes={{128,32}}
  ROOT cpd = f32[128,128] collective-permute-done(cps)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* cps = FindInstruction(module.get(), "cps");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(cps);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{1, {}}, {1}}));
}

// Verifies that collective-permute-start expects no aliasing.
TEST_F(GetInPlaceInputOutputPairsTest, CollectivePermuteStartNotInplace) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[128,32] parameter(0)
  p1 = f32[128,128] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  cps = ((f32[128,32], f32[128,128], s32[], s32[]),
      (f32[128,32], f32[128,128], s32[], s32[]))
      collective-permute-start(p0, p1, p2, p3), source_target_pairs={{0,1}}
  ROOT cpd = (f32[128,32], f32[128,128], s32[], s32[])
      collective-permute-done(cps)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* cps = FindInstruction(module.get(), "cps");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(cps);

  EXPECT_TRUE(pairs.empty());
}

// Verifies in-place behavior for custom-call with output_to_operand_aliasing.
TEST_F(GetInPlaceInputOutputPairsTest, CustomCallAliasing) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = (f32[2,3], f32[2,3]) parameter(0)
  p1 = f32[2,3] parameter(1)
  ROOT cc = (f32[2,3]) custom-call(p0, p1), custom_call_target="foo",
      output_to_operand_aliasing={{0}: (0, {1})}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* cc = FindInstruction(module.get(), "cc");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(cc);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{0, {1}}, {0}}));
}

// Verifies in-place behavior for set-dimension-size.
TEST_F(GetInPlaceInputOutputPairsTest, SetDimensionSize) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[10, 20] parameter(0)
  p1 = s32[] parameter(1)
  ROOT sds = f32[10,<=20] set-dimension-size(p0, p1), dimensions={1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* sds = FindInstruction(module.get(), "sds");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(sds);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{0, {}}, {}}));
}

// Verifies in-place behavior for collective-permute with tuple output.
TEST_F(GetInPlaceInputOutputPairsTest, CollectivePermuteInplaceTuple) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = (f32[128,32], f32[128,32]) parameter(0)
  p1 = (f32[128,128], f32[128,128]) parameter(1)
  p2 = ((s32[], s32[]), (s32[], s32[])) parameter(2)
  p3 = ((s32[], s32[]), (s32[], s32[])) parameter(3)
  ROOT cp = (f32[128,128], f32[128,128])
      collective-permute(p0, p1, p2, p3), source_target_pairs={{0,1}},
      slice_sizes={{128,32}, {128,32}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* cp = FindInstruction(module.get(), "cp");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(cp);

  EXPECT_THAT(
      pairs,
      ElementsAre(
          std::pair<HloOperandIndex, ShapeIndex>{HloOperandIndex{1, {}}, {}},
          std::pair<HloOperandIndex, ShapeIndex>{HloOperandIndex{1, {0}}, {0}},
          std::pair<HloOperandIndex, ShapeIndex>{HloOperandIndex{1, {1}},
                                                 {1}}));
}

// Verifies in-place behavior for collective-permute with non-tuple output.
TEST_F(GetInPlaceInputOutputPairsTest, CollectivePermuteInplaceNonTuple) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[128,32] parameter(0)
  p1 = f32[128,128] parameter(1)
  p2 = (s32[], s32[]) parameter(2)
  p3 = (s32[], s32[]) parameter(3)
  ROOT cp = f32[128,128] collective-permute(p0, p1, p2, p3),
      source_target_pairs={{0,1}}, slice_sizes={{128,32}}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* cp = FindInstruction(module.get(), "cp");

  auto pairs = alias_info_.GetInPlaceInputOutputPairs(cp);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{1, {}}, {}}));
}

class CustomAliasInfo : public AliasInfo {
 public:
  std::optional<std::vector<std::pair<HloOperandIndex, ShapeIndex>>>
  GetNonDefaultInPlaceInputOutputPairs(
      const HloInstruction* user) const override {
    if (user->opcode() == HloOpcode::kAdd) {
      return std::vector<std::pair<HloOperandIndex, ShapeIndex>>{
          {HloOperandIndex{1, {}}, {}}};
    }
    return std::nullopt;
  }
};

// Verifies that a derived AliasInfo can specify non-default pairs.
TEST_F(GetInPlaceInputOutputPairsTest, GetNonDefaultInPlacePairs) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  ROOT add = f32[10] add(p0, p1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* add = FindInstruction(module.get(), "add");

  CustomAliasInfo custom_alias_info;
  auto pairs = custom_alias_info.GetInPlaceInputOutputPairs(add);

  EXPECT_THAT(pairs, ElementsAre(std::pair<HloOperandIndex, ShapeIndex>{
                         HloOperandIndex{1, {}}, {}}));
}

class AliasInfoTest : public HloHardwareIndependentTestBase {};

// Verifies that FollowTupleIndirection correctly identifies the source of a
// tuple element, even if the tuple elements were swapped.
TEST_F(AliasInfoTest, FollowTupleIndirectionBasic) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[2,3] parameter(0)
  p1 = f32[2,3] parameter(1)
  tuple = (f32[2,3], f32[2,3]) tuple(p0, p1)
  gte0 = f32[2,3] get-tuple-element(tuple), index=0
  gte1 = f32[2,3] get-tuple-element(tuple), index=1
  ROOT root = (f32[2,3], f32[2,3]) tuple(gte1, gte0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* root = FindInstruction(module.get(), "root");
  const HloInstruction* tuple = FindInstruction(module.get(), "tuple");

  auto result = FollowTupleIndirection(root, {0});
  EXPECT_EQ(result.first, tuple);
  EXPECT_THAT(result.second, ElementsAre(1));

  result = FollowTupleIndirection(root, {1});
  EXPECT_EQ(result.first, tuple);
  EXPECT_THAT(result.second, ElementsAre(0));
}

// Verifies that FollowTupleIndirection correctly follows a nested tuple
// structure exactly as described in the documentation.
TEST_F(AliasInfoTest, FollowTupleIndirectionAsDescribed) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = (f32[2,3], f32[2,3]) parameter(0)
  p1 = f32[2,3] parameter(1)
  foo = f32[2,3] get-tuple-element(p0), index=0
  ROOT bar = (f32[2,3], f32[2,3]) tuple(p1, foo)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* bar = FindInstruction(module.get(), "bar");
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");

  auto result = FollowTupleIndirection(bar, {1});
  EXPECT_EQ(result.first, p0);
  EXPECT_THAT(result.second, ElementsAre(0));
}

// Verifies the MustAlias functionality for a dynamic-update-slice operation,
// ensuring it aliases with its update base operand but not the update value.
TEST_F(AliasInfoTest, MustAliasTest) {
  const char* const kHlo = R"(
HloModule test

ENTRY main {
  p0 = f32[10,10] parameter(0)
  p1 = f32[2,2] parameter(1)
  i0 = s32[] constant(0)
  i1 = s32[] constant(0)
  ROOT dus = f32[10,10] dynamic-update-slice(p0, p1, i0, i1)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  const HloInstruction* dus = FindInstruction(module.get(), "dus");
  const HloInstruction* p0 = FindInstruction(module.get(), "p0");
  const HloInstruction* p1 = FindInstruction(module.get(), "p1");

  AliasInfo alias_info;

  EXPECT_TRUE(alias_info.MustAlias(p0, {}, dus, {}));
  EXPECT_FALSE(alias_info.MustAlias(p1, {}, dus, {}));
}

}  // namespace
}  // namespace xla
