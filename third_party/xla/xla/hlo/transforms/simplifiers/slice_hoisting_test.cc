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

#include "xla/hlo/transforms/simplifiers/slice_hoisting.h"

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class HloSliceHoistingTest : public HloHardwareIndependentTestBase {};

TEST_F(HloSliceHoistingTest, HoistSlices) {
  const char* const hlo_string = R"(
    HloModule m, is_scheduled=true

    ENTRY e {
      p0 = f32[512,256] parameter(0)
      c1 = f32[] constant(1)

      bcast1 = f32[512,256] broadcast(c1), dimensions={}
      mul = f32[512,256] multiply(bcast1, bcast1)
      s00 = f32[4,4] slice(f32[512,256] p0), slice={[0:4], [0:4]}
      add = f32[512,256] add(bcast1, bcast1)
      s01 = f32[2,2] slice(f32[4,4] s00), slice={[0:2], [0:2]}

      ROOT tuple = (f32[4,4], f32[2,2], f32[512,256]) tuple(s00, s01, mul)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  SliceHoisting pass;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, pass.Run(module.get(), {}));
  EXPECT_TRUE(changed);
  const std::vector<HloInstruction*>& sequence =
      module->schedule().sequence(module->entry_computation()).instructions();

  absl::flat_hash_map<std::string, const HloInstruction*> instructions_by_name;
  for (const HloInstruction* instruction : sequence) {
    instructions_by_name[instruction->name()] = instruction;
  }

  auto index = [&](absl::string_view name) -> size_t {
    const HloInstruction* instruction = instructions_by_name.at(name);
    return std::distance(sequence.begin(), absl::c_find(sequence, instruction));
  };

  std::vector<size_t> indices = {
      index("p0"),     index("s00"), index("s01"), index("c1"),
      index("bcast1"), index("mul"), index("add"),
  };

  EXPECT_TRUE(absl::c_is_sorted(indices));
}

}  // namespace
}  // namespace xla
