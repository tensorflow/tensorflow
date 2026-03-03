/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/collectives/collective_combiner_annotator.h"

#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/hlo_module_config.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {
namespace {

using ::testing::Optional;

class CollectiveCombinerAnnotatorTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<bool> RunCollectiveCombinerAnnotator(
      HloModule* module, int64_t device_memory_size = 2000000000) {
    int pointer_size = 4;
    stream_executor::DeviceDescription device_info;
    device_info.set_device_memory_size(device_memory_size);
    GpuAliasInfo alias_info(device_info);
    return RunHloPass(
        CollectiveCombinerAnnotator(std::move(device_info), &alias_info,
                                    pointer_size, &mlir_context_),
        module);
  }
  mlir::MLIRContext mlir_context_;
};

TEST_F(CollectiveCombinerAnnotatorTest, SynchronousCollectivesNoOverlap) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
        p0 = f16[] parameter(0)
        p1 = f16[] parameter(1)
        ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
        p0 = f16[10000000]{0} parameter(0)
        p1 = f16[10000000]{0} parameter(1)
        ar0 = f16[10000000]{0} all-reduce(p0), replica_groups={}, to_apply=add
        ar1 = f16[10000000]{0} all-reduce(p1), replica_groups={}, to_apply=add
        ROOT result = tuple(ar0, ar1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunCollectiveCombinerAnnotator(module.get()),
              absl_testing::IsOkAndHolds(true));
  const HloInstruction* ar0 =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_TRUE(IsCombinableSyncCollective(*ar0));
  const HloInstruction* ar1 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(IsCombinableSyncCollective(*ar1));
}

TEST_F(CollectiveCombinerAnnotatorTest, SynchronousCollectivesWithOverlap) {
  // Expected schedule:
  // ------------------
  // c0 –> ar0
  //       c1 –> ar1
  // ------------------
  absl::string_view kHloText = R"(
    HloModule m

    add {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
      p0 = f16[10000000]{0} parameter(0)
      p1 = f16[10000000]{0} parameter(1)

      c0 = f16[10000000]{0} copy(p0)
      c1 = f16[10000000]{0} copy(p1)

      ar0 = f16[10000000]{0} all-reduce(c0), replica_groups={}, to_apply=add
      ar1 = f16[10000000]{0} all-reduce(c1), replica_groups={}, to_apply=add

      ROOT result = tuple(ar0, ar1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_THAT(RunCollectiveCombinerAnnotator(module.get()),
              absl_testing::IsOkAndHolds(true));
  const HloInstruction* ar0 =
      module->entry_computation()->root_instruction()->operand(0);
  EXPECT_FALSE(IsCombinableSyncCollective(*ar0));
  const HloInstruction* ar1 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(IsCombinableSyncCollective(*ar1));
}

TEST_F(CollectiveCombinerAnnotatorTest,
       ContainsCombinableSyncCollectiveReturnFalseForNonAnnotatedCollectives) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
      p0 = f16[] parameter(0)
      p1 = f16[] parameter(1)
      ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
      p0 = f16[10000000]{0} parameter(0)
      ROOT result = f16[10000000]{0} all-reduce(p0), replica_groups={}, to_apply=add
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_FALSE(ContainsCombinableSyncCollective(*module));
}

TEST_F(CollectiveCombinerAnnotatorTest,
       ContainsCombinableSyncCollectiveReturnTRUEForAnnotatedCollectives) {
  absl::string_view kHloText = R"(
    HloModule m

    add {
    p0 = f16[] parameter(0)
    p1 = f16[] parameter(1)
    ROOT add = f16[] add(p0, p1)
    }

    ENTRY main {
    p0 = f16[10000000]{0} parameter(0)
    ROOT result = f16[10000000]{0} all-reduce(p0), replica_groups={}, to_apply=add,
      frontend_attributes={sync_collective="true"}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));
  EXPECT_TRUE(ContainsCombinableSyncCollective(*module));
}

TEST_F(CollectiveCombinerAnnotatorTest,
       SuggestedCombinerThresholddFromDeviceInfo) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY ar {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)

      ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
        custom_call_target="__cublas$gemm"
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  // device size = 20000 bytes
  // slop factor = 0.95
  // peak memory = parameters + output = (2*32*32 + 32*32) * 4 bytes = 12288
  // suggested thresholds = device size * slop factor - peak memory
  EXPECT_THAT(RunCollectiveCombinerAnnotator(module.get(),
                                             /*device_memory_size=*/20000),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(SuggestedCombinerThreshold(*module), Optional(6712L));
}

TEST_F(CollectiveCombinerAnnotatorTest,
       SuggestedCombinerThresholddFromModuleConfig) {
  absl::string_view kHloText = R"(
    HloModule m

    ENTRY ar {
      p0 = f32[32,32] parameter(0)
      p1 = f32[32,32] parameter(1)

      ROOT _ = f32[32,32]{1,0} custom-call(p0, p1),
        custom_call_target="__cublas$gemm"
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.set_device_memory_size(20000);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kHloText, config));

  // device size = 20000 bytes
  // slop factor = 0.95
  // peak memory = parameters + output = (2*32*32 + 32*32) * 4 bytes = 12288
  // suggested thresholds = device size * slop factor - peak memory
  EXPECT_THAT(RunCollectiveCombinerAnnotator(module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_THAT(SuggestedCombinerThreshold(*module), Optional(6712L));
}

}  // namespace
}  // namespace xla::gpu
