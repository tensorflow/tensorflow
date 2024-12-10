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

#include <cstdint>
#include <string>

#include "testing/base/public/malloc_counter.h"
#include "absl/strings/str_format.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/transforms/simplifiers/hlo_constant_folding.h"
#include "xla/test.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using HloConstantFoldingTest = HloHardwareIndependentTestBase;
TEST_F(HloConstantFoldingTest, PeakHeapTest) {
  constexpr int kNumAdds = 200;

  // Create an HLO graph that is fully constant foldable and contains a chain
  // of kNumAdds operations.
  std::string mod_str = R"(
    add {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      ROOT add = s32[] add(p0, p1)
    }

    ENTRY main.9 {
      c.0 = s32[] constant(1)
      bc = s32[1,1024,1024] broadcast(c.0), dimensions={}
      zero = s32[] constant(0)
      red = s32[1024,1024] reduce(bc, zero), dimensions={0}, to_apply=add
      add.0 = s32[1024,1024] add(red, red)

   )";

  for (int i = 1; i <= kNumAdds; i++) {
    mod_str +=
        absl::StrFormat("add.%d = s32[1024,1024] add(red, add.%d)\n", i, i - 1);
  }
  mod_str += absl::StrFormat(R"(
      ROOT end = s32[] reduce(add.%d, zero), dimensions={0,1}, to_apply=add
    }
   )",
                             kNumAdds);

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(mod_str));
  HloConstantFolding const_fold;

  // Measure the peah heap growth of the constant folding pass.
  ::testing::MallocCounter mc(::testing::MallocCounter::THIS_THREAD_ONLY);
  TF_ASSERT_OK_AND_ASSIGN(bool result, RunHloPass(&const_fold, module.get()));
  EXPECT_TRUE(result);

  // The limit below is based on a few experimental runs plus a large margin to
  // what was actually measured. It does not have to be very precise, because
  // the value without optimizations is > 15X larger.
  constexpr int64_t kExpectedPeakHeapGrowth = 400000000;
  EXPECT_LT(mc.PeakHeapGrowth(), kExpectedPeakHeapGrowth);
}

}  // namespace
}  // namespace xla
