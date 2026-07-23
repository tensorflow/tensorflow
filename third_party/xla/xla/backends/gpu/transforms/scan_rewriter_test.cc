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

#include "xla/backends/gpu/transforms/scan_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using ScanRewriterTest = HloHardwareIndependentTestBase;

TEST_F(ScanRewriterTest, BasicScan) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100] parameter(0)
  p1 = f32[] constant(0.0)
  scan = (f32[100], f32[]) scan(p0, p1), 
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[100] get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // Check that the scan is rewritten.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      ::xla::GmockMatch(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0)));
}

TEST_F(ScanRewriterTest, SingleElementZeroInit) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[1] parameter(0)
  p1 = f32[1] parameter(1)
  add = f32[1] add(p0, p1)
  ROOT tuple = (f32[1], f32[1]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100,1] parameter(0)
  p1 = f32[1] constant({0})
  scan = (f32[100,1], f32[1]) scan(p0, p1),
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[100,1] get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // Check that the scan is rewritten.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      ::xla::GmockMatch(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0)));
}

TEST_F(ScanRewriterTest, AllZeroVectorInit) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[2] parameter(0)
  p1 = f32[2] parameter(1)
  add = f32[2] add(p0, p1)
  ROOT tuple = (f32[2], f32[2]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[2,100]{1,0} parameter(0)
  p1 = f32[2] constant({0.0, 0.0})
  scan = (f32[2,100]{1,0}, f32[2]) scan(p0, p1), 
    dimensions={1}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[2,100]{1,0} get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // An all zero vector init needs no seeding: plain rewrite.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      ::xla::GmockMatch(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0)));
}

TEST_F(ScanRewriterTest, NonZeroInitAppliedAsSeed) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100] parameter(0)
  p1 = f32[] constant(5.0)
  scan = (f32[100], f32[]) scan(p0, p1),
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[100] get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // The seed is added to the CUB result: out = cub_scan(p0) + broadcast(5).
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::xla::GmockMatch(
                  m::Add(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0),
                         m::Broadcast(m::Constant()))));
}

TEST_F(ScanRewriterTest, VectorSeedFromParameter) {
  // The form the SPMD partitioner emits for scan dim sharded scans: the per
  // shard rerun is seeded with the neighbor shard's incoming carry.
  const char* hlo_text = R"(
HloModule module

add {
  p0 = s32[4] parameter(0)
  p1 = s32[4] parameter(1)
  add = s32[4] add(p0, p1)
  ROOT tuple = (s32[4], s32[4]) tuple(add, add)
}

ENTRY entry {
  p0 = s32[4,32]{1,0} parameter(0)
  seed = s32[4] parameter(1)
  scan = (s32[4,32]{1,0}, s32[4]) scan(p0, seed),
    dimensions={1}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = s32[4,32]{1,0} get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::xla::GmockMatch(
                  m::Add(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0),
                         m::Broadcast(m::Parameter(1)))));
}

TEST_F(ScanRewriterTest, LiveFinalCarry) {
  // The form the SPMD partitioner emits for the shard totals: only the
  // final carry is read. It is the last output element of the inclusive
  // scan.
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100] parameter(0)
  p1 = f32[] constant(0.0)
  scan = (f32[100], f32[]) scan(p0, p1),
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[] get-tuple-element(scan), index=1
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // The carry user is fed a slice of the CUB output directly.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::xla::GmockMatch(m::Reshape(m::Slice(
                  m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0)))));
  const HloInstruction* slice =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_EQ(slice->slice_starts(0), 99);
  EXPECT_EQ(slice->slice_limits(0), 100);
}

TEST_F(ScanRewriterTest, ReverseSeededScanLiveCarry) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100] parameter(0)
  seed = f32[] parameter(1)
  scan = (f32[100], f32[]) scan(p0, seed),
    dimensions={0}, num_carries=1, is_reverse=true, is_associative=true, to_apply=add
  out = f32[100] get-tuple-element(scan), index=0
  carry = f32[] get-tuple-element(scan), index=1
  broadcast = f32[100] broadcast(carry), dimensions={}
  ROOT root = f32[100] add(out, broadcast)
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // A reverse scan's final carry is its first output element; both users
  // share the seed applied output.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root,
              ::xla::GmockMatch(m::Add(
                  m::Add(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0),
                         m::Broadcast(m::Parameter(1))),
                  m::Broadcast(m::Reshape(m::Slice(m::Add(
                      m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0),
                      m::Broadcast(m::Parameter(1)))))))));
  const HloInstruction* slice = root->operand(1)->operand(0)->operand(0);
  EXPECT_EQ(slice->slice_starts(0), 0);
  EXPECT_EQ(slice->slice_limits(0), 1);
  EXPECT_EQ(slice->operand(0), root->operand(0));
}

TEST_F(ScanRewriterTest, DeadNonGetTupleElementUser) {
  // A dead non get-tuple-element user passes IsStandardAssociativeScan; the
  // carry liveness walk must not assume every user is a get-tuple-element.
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[100] parameter(0)
  p1 = f32[] constant(0.0)
  scan = (f32[100], f32[]) scan(p0, p1),
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  dead = ((f32[100], f32[])) tuple(scan)
  ROOT root = f32[100] get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      ::xla::GmockMatch(m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0)));
}

TEST_F(ScanRewriterTest, SkipZeroLengthScanDimension) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  add = f32[] add(p0, p1)
  ROOT tuple = (f32[], f32[]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[0] parameter(0)
  p1 = f32[] constant(0.0)
  scan = (f32[0], f32[]) scan(p0, p1),
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[0] get-tuple-element(scan), index=0
}
)";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // Nothing to scan, and no boundary element for the carry: not rewritten.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::xla::GmockMatch(
                  m::GetTupleElement(m::Op().WithOpcode(HloOpcode::kScan), 0)));
}

TEST_F(ScanRewriterTest, SkipNonMinorScanDimension) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  add = f32[10] add(p0, p1)
  ROOT tuple = (f32[10], f32[10]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[10,10]{1,0} parameter(0)
  scalar_zero = f32[] constant(0.0)
  p1 = f32[10]{0} broadcast(scalar_zero), dimensions={}
  scan = (f32[10,10]{1,0}, f32[10]{0}) scan(p0, p1), 
    dimensions={0}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[10,10]{1,0} get-tuple-element(scan), index=0
}
)";
  // scan_dim is 0. layout is {1,0} so 0 is major. vector_length > 1.
  // It should be skipped.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // Check that the scan is NOT rewritten.
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              ::xla::GmockMatch(m::GetTupleElement(m::Op(), 0)));
}

TEST_F(ScanRewriterTest, AcceptMinorScanDimension) {
  const char* hlo_text = R"(
HloModule module

add {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  add = f32[10] add(p0, p1)
  ROOT tuple = (f32[10], f32[10]) tuple(add, add)
}

ENTRY entry {
  p0 = f32[10,10]{1,0} parameter(0)
  scalar_zero = f32[] constant(0.0)
  p1 = f32[10]{0} broadcast(scalar_zero), dimensions={}
  scan = (f32[10,10]{1,0}, f32[10]{0}) scan(p0, p1), 
    dimensions={1}, num_carries=1, is_associative=true, to_apply=add
  ROOT root = f32[10,10]{1,0} get-tuple-element(scan), index=0
}
)";
  // scan_dim is 1. layout is {1,0} so 1 is minor. vector_length is 1.
  // It should be rewritten.
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  ScanRewriter pass;
  ASSERT_OK(pass.Run(module.get()));

  // Check that the scan is rewritten and is layout-constrained.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, ::xla::GmockMatch(
                        m::GetTupleElement(m::CustomCall(m::Parameter(0)), 0)));

  auto* custom_call = Cast<HloCustomCallInstruction>(root->mutable_operand(0));
  EXPECT_TRUE(custom_call->layout_constrained());
  EXPECT_EQ(
      custom_call->operand_shapes_with_layout()[0].layout().minor_to_major(0),
      1);
}

}  // namespace
}  // namespace xla::gpu
