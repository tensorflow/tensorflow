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

#include "xla/backends/gpu/transforms/constant_fill_copy_rewriter.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/copy_fusion.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla::gpu {
namespace {

constexpr char kHlo[] = R"(
HloModule test

fill_body {
  value = f32[] constant(7)
  ROOT broadcast = f32[512,512]{1,0} broadcast(value), dimensions={}
}

ENTRY main {
  predecessor = f32[] constant(1)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  ROOT copy = f32[512,512]{1,0} copy(fill)
}
)";

constexpr absl::string_view kRootCopy =
    "ROOT copy = f32[512,512]{1,0} copy(fill)";

using ConstantFillCopyRewriterTest = HloHardwareIndependentTestBase;

// Checks that `fill` is a fresh zero-operand fill equivalent to `original`.
void ExpectRematerializedFill(const HloInstruction* fill,
                              const HloInstruction* original) {
  EXPECT_NE(fill, original);
  ASSERT_EQ(fill->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(fill->operand_count(), 0);
  EXPECT_EQ(fill->shape(), original->shape());
  EXPECT_EQ(*fill->fused_instructions_computation(),
            *original->fused_instructions_computation());
}

TEST_F(ConstantFillCopyRewriterTest, ReplacesSoleCopyWithTheFillItself) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  HloComputation* entry = module->entry_computation();
  const HloInstruction* fill = entry->GetInstructionWithName("fill");
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  // The copy was the fill's only use, so the fill stands in for it directly and
  // nothing is cloned.
  EXPECT_EQ(entry->root_instruction(), fill);
  EXPECT_EQ(entry->GetInstructionWithName("copy"), nullptr);
  EXPECT_EQ(module->computation_count(), 2);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest, RematerializesFillForItsOwnConsumer) {
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRootCopy, R"(
  copy = f32[512,512]{1,0} copy(fill)
  early = f32[512,512]{1,0} negate(copy)
  ROOT late = (f32[512,512]{1,0}, f32[512,512]{1,0}) tuple(early, fill))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  const HloInstruction* fill = entry->GetInstructionWithName("fill");
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ExpectRematerializedFill(entry->GetInstructionWithName("early")->operand(0),
                           fill);
  // The original fill keeps its late consumer and its fusion body.
  EXPECT_EQ(entry->root_instruction()->operand(1), fill);
  EXPECT_EQ(entry->GetInstructionWithName("copy"), nullptr);
  EXPECT_EQ(module->computation_count(), 3);

  // The following pipeline pass must leave the independent fills intact.
  const auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CopyFusion copy_fusion(device);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&copy_fusion, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest,
       KeepsCopyWhoseConsumerReceivesEveryOtherUse) {
  // copy0 feeds early on its own, while copy1 and the fill both feed late.
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRootCopy, R"(
  copy0 = f32[512,512]{1,0} copy(fill)
  early = f32[512,512]{1,0} negate(copy0)
  copy1 = f32[512,512]{1,0} copy(fill)
  ROOT late = (f32[512,512]{1,0}, f32[512,512]{1,0}, f32[512,512]{1,0})
      tuple(early, fill, copy1))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  const HloInstruction* fill = entry->GetInstructionWithName("fill");
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ExpectRematerializedFill(entry->GetInstructionWithName("early")->operand(0),
                           fill);
  // Once copy0 has its own fill, every remaining use of the fill feeds late,
  // so copy1 stays a copy for CopyFusion to fold into the fill.
  const HloInstruction* late = entry->root_instruction();
  EXPECT_EQ(late->operand(1), fill);
  EXPECT_EQ(late->operand(2), entry->GetInstructionWithName("copy1"));
  EXPECT_EQ(late->operand(2)->operand(0), fill);

  const auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CopyFusion copy_fusion(device);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&copy_fusion, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_EQ(late->operand(1)->opcode(), HloOpcode::kGetTupleElement);
  ASSERT_EQ(late->operand(2)->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(late->operand(1)->operand(0), late->operand(2)->operand(0));
  EXPECT_TRUE(late->operand(1)->operand(0)->IsMultiOutputFusion());
}

TEST_F(ConstantFillCopyRewriterTest, LooksThroughBitcastLikeCopyFusion) {
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRootCopy, R"(
  bitcast = f32[262144]{0} bitcast(fill)
  copy = f32[262144]{0} copy(bitcast)
  early = f32[262144]{0} negate(copy)
  ROOT late = (f32[262144]{0}, f32[512,512]{1,0}) tuple(early, fill))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  const HloInstruction* fill = entry->GetInstructionWithName("fill");
  const HloInstruction* bitcast = entry->GetInstructionWithName("bitcast");
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  // The copy becomes a bitcast of a fresh fill and the old bitcast dies with
  // it, leaving predecessor, fill, the new fill and bitcast, early, and late.
  const HloInstruction* early = entry->GetInstructionWithName("early");
  const HloInstruction* new_bitcast = early->operand(0);
  EXPECT_NE(new_bitcast, bitcast);
  ASSERT_EQ(new_bitcast->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(new_bitcast->shape(), early->shape());
  ExpectRematerializedFill(new_bitcast->operand(0), fill);
  EXPECT_EQ(entry->root_instruction()->operand(1), fill);
  EXPECT_EQ(entry->instruction_count(), 6);
  EXPECT_EQ(module->computation_count(), 3);

  const auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CopyFusion copy_fusion(device);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&copy_fusion, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest, ReplacesSoleCopyOfBitcastWithTheBitcast) {
  std::string hlo = absl::StrReplaceAll(kHlo, {{kRootCopy, R"(
  bitcast = f32[262144]{0} bitcast(fill)
  ROOT copy = f32[262144]{0} copy(bitcast))"}});
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloComputation* entry = module->entry_computation();
  const HloInstruction* bitcast = entry->GetInstructionWithName("bitcast");
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(entry->root_instruction(), bitcast);
  EXPECT_EQ(bitcast->operand(0), entry->GetInstructionWithName("fill"));
  EXPECT_EQ(module->computation_count(), 2);
}

TEST_F(ConstantFillCopyRewriterTest, RespectsExecutionThreads) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       ConstantFillCopyRewriter().Run(module.get(), {"other"}));
  EXPECT_FALSE(changed);
}

TEST_F(ConstantFillCopyRewriterTest, ReducesScheduledPeakMemory) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule test
sum {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}
fill_body {
  zero = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(zero), dimensions={}
}
ENTRY main {
  update = f32[1,512]{1,0} parameter(0)
  zero = s32[] constant(0)
  one = s32[] constant(1)
  initial_sum = f32[] constant(0)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  copy = f32[512,512]{1,0} copy(fill)
  early = f32[512,512]{1,0} dynamic-update-slice(copy, update, zero, zero)
  early_sum = f32[] reduce(early, initial_sum), dimensions={0,1}, to_apply=sum
  middle = f32[1024,1024]{1,0} broadcast(early_sum), dimensions={}
  middle_sum = f32[] reduce(middle, initial_sum), dimensions={0,1}, to_apply=sum
  late_update = f32[1,512]{1,0} broadcast(middle_sum), dimensions={}
  ROOT late = f32[512,512]{1,0}
      dynamic-update-slice(fill, late_update, one, zero)
})"));
  auto baseline = module->Clone();
  const auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  GpuAliasInfo alias_info(device);
  CopyFusion copy_fusion(device);
  ASSERT_OK_AND_ASSIGN(bool baseline_changed,
                       RunHloPass(&copy_fusion, baseline.get()));
  EXPECT_TRUE(baseline_changed);
  auto size = [](const BufferValue& buffer) {
    return ShapeUtil::ByteSizeOf(buffer.shape(), /*pointer_size=*/8);
  };
  int64_t baseline_peak;
  ASSERT_OK_AND_ASSIGN(
      auto baseline_schedule,
      ScheduleModule(baseline.get(), &alias_info, size, {}, &baseline_peak));
  ASSERT_OK(baseline_schedule.Verify());

  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_TRUE(changed);
  ASSERT_OK_AND_ASSIGN(changed, RunHloPass(&copy_fusion, module.get()));
  EXPECT_FALSE(changed);
  int64_t peak;
  ASSERT_OK_AND_ASSIGN(auto schedule, ScheduleModule(module.get(), &alias_info,
                                                     size, {}, &peak));
  ASSERT_OK(schedule.Verify());
  // The original 1 MiB fill no longer needs to span the 4 MiB middle buffer.
  EXPECT_LT(peak, baseline_peak);
}

TEST_F(ConstantFillCopyRewriterTest, LeavesCopiesInsideWhileBodyAlone) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
HloModule test
fill_body {
  zero = f32[] constant(0)
  ROOT broadcast = f32[512,512]{1,0} broadcast(zero), dimensions={}
}
condition {
  p = f32[512,512]{1,0} parameter(0)
  ROOT stop = pred[] constant(false)
}
body {
  p = f32[512,512]{1,0} parameter(0)
  fill = f32[512,512]{1,0} fusion(), kind=kLoop, calls=fill_body
  ROOT copy = f32[512,512]{1,0} copy(fill)
}
ENTRY main {
  p = f32[512,512]{1,0} parameter(0)
  ROOT loop = f32[512,512]{1,0} while(p), condition=condition, body=body
})"));
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
}

struct RejectedCopy {
  const char* name;
  std::vector<std::pair<absl::string_view, absl::string_view>> replacements;
};

class RejectedConstantFillCopyTest
    : public ConstantFillCopyRewriterTest,
      public ::testing::WithParamInterface<RejectedCopy> {};

TEST_P(RejectedConstantFillCopyTest, LeavesCopyUnchanged) {
  const RejectedCopy& test = GetParam();
  std::string hlo = absl::StrReplaceAll(kHlo, test.replacements);
  ASSERT_NE(hlo, kHlo);
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  std::string before = module->ToString();
  ConstantFillCopyRewriter pass;
  ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&pass, module.get()));
  EXPECT_FALSE(changed);
  EXPECT_EQ(module->ToString(), before);
}

INSTANTIATE_TEST_SUITE_P(
    Safeguards, RejectedConstantFillCopyTest,
    ::testing::Values(
        RejectedCopy{"SmallFill", {{"512", "256"}}},
        RejectedCopy{"LayoutChange",
                     {{"ROOT copy = f32[512,512]{1,0}",
                       "ROOT copy = f32[512,512]{0,1}"}}},
        RejectedCopy{
            "ParameterFill",
            {{"value = f32[] constant(7)", "value = f32[] parameter(0)"},
             {"fusion()", "fusion(predecessor)"}}},
        RejectedCopy{"ComputedFill",
                     {{"ROOT broadcast",
                       "negated = f32[] negate(value)\n"
                       "  ROOT broadcast"},
                      {"broadcast(value)", "broadcast(negated)"}}},
        RejectedCopy{"NonscalarLiteral",
                     {{"f32[] constant(7)", "f32[1] constant({7})"},
                      {"512,512", "1,262144"},
                      {"dimensions={}", "dimensions={0}"}}},
        RejectedCopy{
            "CopyControlDependency",
            {{"copy(fill)", "copy(fill), control-predecessors={predecessor}"}}},
        RejectedCopy{"FillControlDependency",
                     {{"calls=fill_body",
                       "calls=fill_body, control-predecessors={predecessor}"}}},
        RejectedCopy{"CopySharding",
                     {{"copy(fill)", "copy(fill), sharding={replicated}"}}},
        RejectedCopy{
            "FillSharding",
            {{"calls=fill_body", "calls=fill_body, sharding={replicated}"}}},
        RejectedCopy{"DeadCopy", {{kRootCopy, R"(
  copy = f32[512,512]{1,0} copy(fill)
  ROOT other = f32[512,512]{1,0} negate(fill))"}}},
        // Every use of the fill feeds one consumer, so all of those buffers
        // are live there regardless of how they are produced.
        RejectedCopy{"SharedConsumer", {{kRootCopy, R"(
  copy0 = f32[512,512]{1,0} copy(fill)
  copy1 = f32[512,512]{1,0} copy(fill)
  ROOT result = (f32[512,512]{1,0}, f32[512,512]{1,0}, f32[512,512]{1,0})
      tuple(fill, copy0, copy1))"}}},
        RejectedCopy{"SharedConsumerOfCopiesOnly", {{kRootCopy, R"(
  copy0 = f32[512,512]{1,0} copy(fill)
  copy1 = f32[512,512]{1,0} copy(fill)
  ROOT result = (f32[512,512]{1,0}, f32[512,512]{1,0}) tuple(copy0, copy1))"}}},
        RejectedCopy{"SharedBitcast", {{kRootCopy, R"(
  bitcast = f32[512,512]{1,0} bitcast(fill)
  copy = f32[512,512]{1,0} copy(bitcast)
  other = f32[512,512]{1,0} negate(bitcast)
  ROOT result = (f32[512,512]{1,0}, f32[512,512]{1,0}) tuple(copy, other))"}}},
        RejectedCopy{"BitcastChangesBitwidth", {{kRootCopy, R"(
  bitcast = f16[512,1024]{1,0} bitcast(fill)
  ROOT copy = f16[512,1024]{1,0} copy(bitcast))"}}}),
    [](const ::testing::TestParamInfo<RejectedCopy>& info) {
      return info.param.name;
    });

}  // namespace
}  // namespace xla::gpu
