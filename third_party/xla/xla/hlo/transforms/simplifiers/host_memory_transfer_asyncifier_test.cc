/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/host_memory_transfer_asyncifier.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = ::xla::match;

class HostMemoryTransferAsyncifierTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunAsyncifier(absl::string_view hlo_string) {
    TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnVerifiedModule(hlo_string));
    TF_ASSIGN_OR_RETURN(bool changed, RunAsyncifier(module.get()));
    return changed;
  }

  absl::StatusOr<bool> RunAsyncifier(HloModule* module) {
    TF_EXPECT_OK(verifier().Run(module).status());
    if (module->has_schedule()) {
      return absl::InternalError("Expected a non-scheduled module");
    }

    HostMemoryTransferAsyncifier asyncifier(kHostMemorySpaceColor);
    return asyncifier.Run(module);
  }

 private:
  static constexpr int64_t kHostMemorySpaceColor{5};
};

// =============================DynamicUpdateSlice==============================

TEST_F(HostMemoryTransferAsyncifierTest, DynamicUpdateSliceFromHostToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_operand = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  host_update = f32[1,1,1]{2,1,0:T(2,128)S(5)} parameter(1)
  constant_0 = s32[] constant(0)
  ROOT dynamic-update-slice = f32[32,1,1]{2,1,0:T(2,128)S(5)} dynamic-update-slice(host_operand, host_update, constant_0, constant_0, constant_0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular dynamic-update-slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice()));
}

TEST_F(HostMemoryTransferAsyncifierTest, DynamicUpdateSliceFromDeviceToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  operand = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  update = f32[1,1,1]{2,1,0:T(2,128)} parameter(1)
  constant_0 = s32[] constant(0)
  ROOT dynamic-update-slice = f32[32,1,1]{2,1,0:T(2,128)} dynamic-update-slice(operand, update, constant_0, constant_0, constant_0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular dynamic-update-slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice()));
}

TEST_F(HostMemoryTransferAsyncifierTest, DynamicUpdateSliceFromHostToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  operand = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  host_update = f32[1,1,1]{2,1,0:T(2,128)S(5)} parameter(1)
  constant_0 = s32[] constant(0)
  ROOT dynamic-update-slice = f32[32,1,1]{2,1,0:T(2,128)} dynamic-update-slice(operand, host_update, constant_0, constant_0, constant_0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular dynamic-update-slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice()));
}

TEST_F(HostMemoryTransferAsyncifierTest, DynamicUpdateSliceFromDeviceToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_operand = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  update = f32[1,1,1]{2,1,0:T(2,128)} parameter(1)
  constant_0 = s32[] constant(0)
  ROOT dynamic-update-slice = f32[32,1,1]{2,1,0:T(2,128)S(5)} dynamic-update-slice(host_operand, update, constant_0, constant_0, constant_0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_TRUE(changed);
  // dynamic-update-slice should have been converted into an
  // async-dynamic-update-slice.
  HloInstruction* dynamic_update_slice_start;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Op()
                     .WithOpcode(HloOpcode::kAsyncDone)
                     .WithOperand(0, m::Op(&dynamic_update_slice_start)
                                         .WithOpcode(HloOpcode::kAsyncStart))));
  ASSERT_EQ(dynamic_update_slice_start->called_computations().size(), 1);
  HloComputation* async_dynamic_slice_computation =
      dynamic_update_slice_start->called_computations().at(0);
  EXPECT_THAT(async_dynamic_slice_computation->root_instruction(),
              GmockMatch(m::DynamicUpdateSlice()));
}

// ================================DynamicSlice=================================

TEST_F(HostMemoryTransferAsyncifierTest, DynamicSliceFromHostToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_memory = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  constant_0 = s32[] constant(0)
  ROOT dynamic-slice = f32[1,1,1]{2,1,0:T(2,128)S(5)} dynamic-slice(host_memory, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,1,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular dynamic-slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice()));
}

TEST_F(HostMemoryTransferAsyncifierTest, DynamicSliceFromDeviceToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  device = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  constant_0 = s32[] constant(0)
  ROOT dynamic-slice = f32[1,1,1]{2,1,0:T(2,128)} dynamic-slice(device, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,1,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular dynamic-slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice()));
}

TEST_F(HostMemoryTransferAsyncifierTest, DynamicSliceFromDeviceToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  device = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  constant_0 = s32[] constant(0)
  ROOT dynamic-slice = f32[1,1,1]{2,1,0:T(2,128)S(5)} dynamic-slice(device, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,1,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular dynamic-slice
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::DynamicSlice()));
}

TEST_F(HostMemoryTransferAsyncifierTest, DynamicSliceFromHostToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_memory = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  constant_0 = s32[] constant(0)
  ROOT dynamic-slice = f32[1,1,1]{2,1,0:T(2,128)} dynamic-slice(host_memory, constant_0, constant_0, constant_0), dynamic_slice_sizes={1,1,1}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_TRUE(changed);
  // dynamic-slice should have been converted into an async-dynamic-slice.
  HloInstruction* dynamic_slice_start;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Op()
                     .WithOpcode(HloOpcode::kAsyncDone)
                     .WithOperand(0, m::Op(&dynamic_slice_start)
                                         .WithOpcode(HloOpcode::kAsyncStart))));
  ASSERT_EQ(dynamic_slice_start->called_computations().size(), 1);
  HloComputation* async_dynamic_slice_computation =
      dynamic_slice_start->called_computations().at(0);
  EXPECT_THAT(async_dynamic_slice_computation->root_instruction(),
              GmockMatch(m::DynamicSlice()));
}

// ====================================Copy=====================================

TEST_F(HostMemoryTransferAsyncifierTest, CopyFromHostToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_memory = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  ROOT copy = f32[32,1,1]{2,1,0:T(2,128)S(5)} copy(host_memory)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular copy
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy()));
}

TEST_F(HostMemoryTransferAsyncifierTest, CopyFromDeviceToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  device = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  ROOT copy = f32[32,1,1]{2,1,0:T(2,128)} copy(device)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_FALSE(changed);
  // The root instruction should still be a regular copy
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Copy()));
}

// TODO(b/319466176): Once this bug is fixed, enable this test and delete the
// OldCopyFromDeviceToHost test.
TEST_F(HostMemoryTransferAsyncifierTest, DISABLED_CopyFromDeviceToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  device = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  ROOT copy = f32[32,1,1]{2,1,0:T(2,128)S(5)} copy(device)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_TRUE(changed);
  // copy should have been converted into an async-copy.
  HloInstruction* copy_start;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Op()
              .WithOpcode(HloOpcode::kAsyncDone)
              .WithOperand(
                  0, m::Op(&copy_start).WithOpcode(HloOpcode::kAsyncStart))));
  ASSERT_EQ(copy_start->called_computations().size(), 1);
  HloComputation* async_copy_computation =
      copy_start->called_computations().at(0);
  EXPECT_THAT(async_copy_computation->root_instruction(),
              GmockMatch(m::Copy()));
}

TEST_F(HostMemoryTransferAsyncifierTest, OldCopyFromDeviceToHost) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  device = f32[32,1,1]{2,1,0:T(2,128)} parameter(0)
  ROOT copy = f32[32,1,1]{2,1,0:T(2,128)S(5)} copy(device)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_TRUE(changed);
  // copy should have been converted into an async-copy.
  HloInstruction* copy_start;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Op()
              .WithOpcode(HloOpcode::kCopyDone)
              .WithOperand(
                  0, m::Op(&copy_start).WithOpcode(HloOpcode::kCopyStart))));
}

// TODO(b/319466176): Once this bug is fixed, enable this test and delete the
// OldCopyFromHostToDevice test.
TEST_F(HostMemoryTransferAsyncifierTest, DISABLED_CopyFromHostToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_memory = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  ROOT copy = f32[32,1,1]{2,1,0:T(2,128)} copy(host_memory)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_TRUE(changed);
  // copy should have been converted into an async-copy.
  HloInstruction* copy_start;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Op()
              .WithOpcode(HloOpcode::kAsyncDone)
              .WithOperand(
                  0, m::Op(&copy_start).WithOpcode(HloOpcode::kAsyncStart))));
  ASSERT_EQ(copy_start->called_computations().size(), 1);
  HloComputation* async_copy_computation =
      copy_start->called_computations().at(0);
  EXPECT_THAT(async_copy_computation->root_instruction(),
              GmockMatch(m::Copy()));
}

TEST_F(HostMemoryTransferAsyncifierTest, OldCopyFromHostToDevice) {
  const std::string& hlo_string = R"(
HloModule MyModule

ENTRY main {
  host_memory = f32[32,1,1]{2,1,0:T(2,128)S(5)} parameter(0)
  ROOT copy = f32[32,1,1]{2,1,0:T(2,128)} copy(host_memory)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunAsyncifier(module.get()));

  EXPECT_TRUE(changed);
  // copy should have been converted into an async-copy.
  HloInstruction* copy_start;
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(
          m::Op()
              .WithOpcode(HloOpcode::kCopyDone)
              .WithOperand(
                  0, m::Op(&copy_start).WithOpcode(HloOpcode::kCopyStart))));
}

// =============================================================================

}  // namespace

}  // namespace xla
