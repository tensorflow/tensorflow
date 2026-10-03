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

#include "xla/service/scheduler_memory_fencing.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

constexpr int64_t kMebibyte = 1024 * 1024;

int64_t ShapeSize(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/8);
}

class SchedulerMemoryFencingTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunFencing(HloModule* module,
                                  int64_t size_threshold_bytes,
                                  int32_t slack_windows) {
    SchedulerMemoryFencing pass(&ShapeSize, size_threshold_bytes, slack_windows,
                                &alias_info_);
    return pass.Run(module);
  }

  const HloInstruction* Instr(HloModule* module, absl::string_view name) {
    const HloInstruction* instruction =
        module->entry_computation()->GetInstructionWithName(name);
    CHECK_NE(instruction, nullptr) << name;
    return instruction;
  }

  AliasInfo alias_info_;
};

// The last user of the large buffer `big` is `wg`, scheduled inside the first
// collective window. With one window of slack, `wg` must be fenced before the
// start of the second window.
constexpr absl::string_view kLayeredHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  cp1s = (f32[16]{0}, f32[16]{0}) collective-permute-start(p1), source_target_pairs={{0,1},{1,0}}
  wg = f32[1024,1024]{1,0} multiply(big, big)
  cp1d = f32[16]{0} collective-permute-done(cp1s)
  cp2s = (f32[16]{0}, f32[16]{0}) collective-permute-start(cp1d), source_target_pairs={{0,1},{1,0}}
  cp2d = f32[16]{0} collective-permute-done(cp2s)
  ROOT r = f32[16]{0} add(cp2d, cp2d)
})";

TEST_F(SchedulerMemoryFencingTest, FencesLastUserToSlackShiftedStart) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLayeredHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_TRUE(changed);
  // `wg` sits inside window 0 (cp1s..cp1d); slack 1 targets window 1's start.
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "cp2s")));
  // The schedule was not touched and still satisfies the new dependency.
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(SchedulerMemoryFencingTest, ThresholdFiltersSmallBuffers) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLayeredHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunFencing(module.get(), /*size_threshold_bytes=*/100 * kMebibyte,
                 /*slack_windows=*/1));
  EXPECT_FALSE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(), IsEmpty());
}

// `wg` runs before the first window here, so slack 0 fences it to the first
// window's start while slack 1 allows deferring it into the first window.
constexpr absl::string_view kUseBeforeWindowHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  wg = f32[1024,1024]{1,0} multiply(big, big)
  cp1s = (f32[16]{0}, f32[16]{0}) collective-permute-start(p1), source_target_pairs={{0,1},{1,0}}
  cp1d = f32[16]{0} collective-permute-done(cp1s)
  cp2s = (f32[16]{0}, f32[16]{0}) collective-permute-start(cp1d), source_target_pairs={{0,1},{1,0}}
  cp2d = f32[16]{0} collective-permute-done(cp2s)
  ROOT r = f32[16]{0} add(cp2d, cp2d)
})";

TEST_F(SchedulerMemoryFencingTest, SlackZeroFencesToOwnWindowStart) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUseBeforeWindowHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/0));
  EXPECT_TRUE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "cp1s")));
}

TEST_F(SchedulerMemoryFencingTest, SlackOneShiftsTargetByOneWindow) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUseBeforeWindowHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_TRUE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "cp2s")));
}

TEST_F(SchedulerMemoryFencingTest, RejectsNegativeSlack) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kUseBeforeWindowHlo));
  absl::StatusOr<bool> result =
      RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                 /*slack_windows=*/-1);
  ASSERT_FALSE(result.ok());
  EXPECT_THAT(result.status().message(), HasSubstr("slack_windows_ >= 0"));
}

TEST_F(SchedulerMemoryFencingTest, FencesAcrossCopyWindows) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  copy1s = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p1)
  wg = f32[1024,1024]{1,0} multiply(big, big)
  copy1d = f32[16]{0} copy-done(copy1s)
  copy2s = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(copy1d)
  copy2d = f32[16]{0} copy-done(copy2s)
  ROOT r = f32[16]{0} add(copy2d, copy2d)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_TRUE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "copy2s")));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(SchedulerMemoryFencingTest, FencesAcrossSendRecvWindows) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  token0 = token[] after-all()
  big = f32[1024,1024]{1,0} add(p0, p0)
  recv1 = (f32[16]{0}, u32[], token[]) recv(token0), channel_id=1
  wg = f32[1024,1024]{1,0} multiply(big, big)
  recv1d = (f32[16]{0}, token[]) recv-done(recv1), channel_id=1
  recv_token = token[] get-tuple-element(recv1d), index=1
  send2 = (f32[16]{0}, u32[], token[]) send(p1, recv_token), channel_id=2
  ROOT send2d = token[] send-done(send2), channel_id=2
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_TRUE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "send2")));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(SchedulerMemoryFencingTest, FencesAcrossGenericAsyncWindows) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

async_computation {
  p = f32[16]{0} parameter(0)
  ROOT n = f32[16]{0} negate(p)
}

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  call1s = ((f32[16]{0}), f32[16]{0}, u32[]) call-start(p1), to_apply=async_computation
  wg = f32[1024,1024]{1,0} multiply(big, big)
  call1d = f32[16]{0} call-done(call1s)
  call2s = ((f32[16]{0}), f32[16]{0}, u32[]) call-start(call1d), to_apply=async_computation
  call2d = f32[16]{0} call-done(call2s)
  ROOT r = f32[16]{0} add(call2d, call2d)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_TRUE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "call2s")));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(SchedulerMemoryFencingTest, SkipsEdgeImpliedByDataDependencies) {
  // `wg` feeds the second collective's operand chain, so the fence
  // wg -> cp2s is already implied and must not be added.
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  cp1s = (f32[16]{0}, f32[16]{0}) collective-permute-start(p1), source_target_pairs={{0,1},{1,0}}
  wg = f32[1024,1024]{1,0} multiply(big, big)
  cp1d = f32[16]{0} collective-permute-done(cp1s)
  wg_slice = f32[16,1]{1,0} slice(wg), slice={[0:16], [0:1]}
  wg_small = f32[16]{0} reshape(wg_slice)
  sum = f32[16]{0} add(wg_small, cp1d)
  cp2s = (f32[16]{0}, f32[16]{0}) collective-permute-start(sum), source_target_pairs={{0,1},{1,0}}
  cp2d = f32[16]{0} collective-permute-done(cp2s)
  ROOT r = f32[16]{0} add(cp2d, cp2d)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_FALSE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(), IsEmpty());
}

TEST_F(SchedulerMemoryFencingTest, SkipsBuffersLiveOutOfComputation) {
  // `big` is part of the root tuple, so its live range does not end at its
  // last user and it must not be fenced.
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  cp1s = (f32[16]{0}, f32[16]{0}) collective-permute-start(p1), source_target_pairs={{0,1},{1,0}}
  wg = f32[1024,1024]{1,0} multiply(big, big)
  cp1d = f32[16]{0} collective-permute-done(cp1s)
  cp2s = (f32[16]{0}, f32[16]{0}) collective-permute-start(cp1d), source_target_pairs={{0,1},{1,0}}
  cp2d = f32[16]{0} collective-permute-done(cp2s)
  ROOT r = (f32[16]{0}, f32[1024,1024]{1,0}) tuple(cp2d, big)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_FALSE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(), IsEmpty());
}

TEST_F(SchedulerMemoryFencingTest, NoAsyncWindowsLeavesModuleUnchanged) {
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  big = f32[1024,1024]{1,0} add(p0, p0)
  wg = f32[1024,1024]{1,0} multiply(big, big)
  ROOT r = f32[1024,1024]{1,0} subtract(wg, big)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_FALSE(changed);
}

TEST_F(SchedulerMemoryFencingTest, FencesAllUsersOfABuffer) {
  // `big` has two independent consumers; the buffer stays live until both
  // have executed, so both must receive the fence — constraining only the
  // schedule-last one would let the other be deferred arbitrarily far.
  constexpr absl::string_view kHlo = R"(
HloModule m, is_scheduled=true

ENTRY entry {
  p0 = f32[1024,1024]{1,0} parameter(0)
  p1 = f32[16]{0} parameter(1)
  big = f32[1024,1024]{1,0} add(p0, p0)
  cp1s = (f32[16]{0}, f32[16]{0}) collective-permute-start(p1), source_target_pairs={{0,1},{1,0}}
  wg = f32[1024,1024]{1,0} multiply(big, big)
  sibling = f32[1024,1024]{1,0} subtract(big, p0)
  cp1d = f32[16]{0} collective-permute-done(cp1s)
  cp2s = (f32[16]{0}, f32[16]{0}) collective-permute-start(cp1d), source_target_pairs={{0,1},{1,0}}
  cp2d = f32[16]{0} collective-permute-done(cp2s)
  ROOT r = f32[16]{0} add(cp2d, cp2d)
})";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/1));
  EXPECT_TRUE(changed);
  // Both users of `big` are fenced to the same target, not just `sibling`
  // (the last user in the reference schedule).
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "cp2s")));
  EXPECT_THAT(Instr(module.get(), "sibling")->control_successors(),
              UnorderedElementsAre(Instr(module.get(), "cp2s")));
  EXPECT_OK(module->schedule().Verify());
}

TEST_F(SchedulerMemoryFencingTest, SkipsWhenSlackExceedsRemainingWindows) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kLayeredHlo));
  ASSERT_OK_AND_ASSIGN(
      bool changed, RunFencing(module.get(), /*size_threshold_bytes=*/kMebibyte,
                               /*slack_windows=*/2));
  EXPECT_FALSE(changed);
  EXPECT_THAT(Instr(module.get(), "wg")->control_successors(), IsEmpty());
}

}  // namespace
}  // namespace xla
