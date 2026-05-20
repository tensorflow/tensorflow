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

#include "xla/backends/gpu/libraries/cub/cub_sort_utils.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"

namespace xla::gpu {
namespace {

TEST(CubSortUtilsTest, NonSegmentedDoesntChangeScratchSize) {
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/100,
                                                 /*batch_size=*/1),
            100);
}

TEST(CubSortUtilsTest, SegmentedSortAddsSpace) {
  // For batch_size > 1, the scratch size is updated as follows:
  // 1. Aligned to sizeof(int32_t) = 4 boundary.
  // TODO(b/502873525): This alignment logic adds padding even when already
  // aligned. Fix in a follow-up.
  // 2. Offsets space added: (batch_size + 1) * sizeof(int32_t).

  // scratch_size = 0: aligned to 4 -> 4. Offsets: (2+1)*4 = 12. Total = 16.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/0,
                                                 /*batch_size=*/2),
            16);
  // scratch_size = 1: aligned to 4 -> 4. Offsets: 12. Total = 16.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/1,
                                                 /*batch_size=*/2),
            16);
  // scratch_size = 3: aligned to 4 -> 4. Offsets: 12. Total = 16.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/3,
                                                 /*batch_size=*/2),
            16);
  // scratch_size = 4: aligned to 4 -> 8 (always adds padding if %4==0).
  // Offsets: 12. Total = 20.
  EXPECT_EQ(AddSegmentedSortOffsetsToScratchSize(/*scratch_size=*/4,
                                                 /*batch_size=*/2),
            20);
}

TEST(CreateCubSortCustomCallTest, CreatesCorrectCustomCall) {
  absl::string_view hlo = R"hlo(
    HloModule m
    ENTRY main {
      %keys = f32[10] parameter(0)
      ROOT %custom-call = (f32[10]{0}, u8[0]{0})
        custom-call(%keys),
        custom_call_target="dummy_target"
  })hlo";

  auto module_or = ParseAndReturnUnverifiedModule(hlo);
  ASSERT_TRUE(module_or.ok());
  auto module = std::move(module_or).value();
  auto* computation = module->entry_computation();
  auto* custom_call =
      Cast<HloCustomCallInstruction>(computation->root_instruction());

  ASSERT_OK(CreateCubSortCustomCall(custom_call, /*scratch_size=*/100,
                                    /*ffi_target=*/"ffi_sort_target",
                                    /*descending=*/true,
                                    /*batch_size=*/1));

  HloInstruction* root = computation->root_instruction();
  EXPECT_EQ(root->custom_call_target(), "ffi_sort_target");
  EXPECT_EQ(root->shape().tuple_shapes(1).dimensions(0), 100);
  auto* new_cc = Cast<HloCustomCallInstruction>(root);
  EXPECT_EQ(new_cc->api_version(), CustomCallApiVersion::API_VERSION_TYPED_FFI);
  EXPECT_EQ(new_cc->raw_backend_config_string(),
            "{descending = true, batch_size = 1 : i64}");
}

}  // namespace
}  // namespace xla::gpu
