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

#include "xla/hlo/transforms/simplifiers/constant_deferring.h"

#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class HloSchedulingTest : public HloHardwareIndependentTestBase {};

TEST_F(HloSchedulingTest, DeferConstants) {
  const char* const hlo_string = R"(
    HloModule m, is_scheduled=true

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    ENTRY e {
      p0 = f32[1,2,1,512,256] parameter(0)
      c0 = f32[] constant(0)

      c1 = f32[] constant(1)
      bcast1 = f32[1,2,1,512,256] broadcast(c1), dimensions={}
      x.0 = f32[1,2,1,512,256] add(p0, p0)
      y.0 = f32[1,2,1,512,256] multiply(x.0, x.0)
      add1 = f32[1,2,1,512,256] add(y.0, bcast1)

      c2 = f32[] constant(2)
      bcast2 = f32[1,2,1,512,256] broadcast(c2), dimensions={}
      x.1 = f32[1,2,1,512,256] add(add1, add1)
      y.1 = f32[1,2,1,512,256] multiply(x.1, x.1)
      add2 = f32[1,2,1,512,256] add(y.1, bcast2)

      c3 = f32[] constant(3)
      bcast3 = f32[1,2,1,512,256] broadcast(c3), dimensions={}
      x.2 = f32[1,2,1,512,256] add(add2, add2)
      y.2 = f32[1,2,1,512,256] multiply(x.2, x.2)
      add3 = f32[1,2,1,512,256] add(y.2, bcast3)

      c4 = f32[] constant(4)
      bcast4 = f32[1,2,1,512,256] broadcast(c4), dimensions={}
      x.3 = f32[1,2,1,512,256] add(add3, add3)
      y.3 = f32[1,2,1,512,256] multiply(x.3, x.3)
      add4 = f32[1,2,1,512,256] add(y.3, bcast4)

      c5 = f32[] constant(5)
      bcast5 = f32[1,2,1,512,256] broadcast(c5), dimensions={}
      x.4 = f32[1,2,1,512,256] add(add4, add4)
      y.4 = f32[1,2,1,512,256] multiply(x.4, x.4)
      add5 = f32[1,2,1,512,256] add(y.4, bcast5)

      r1 = f32[1,2] reduce(add1, c0), dimensions={2,3,4}, to_apply=add
      r2 = f32[1,2] reduce(add2, c0), dimensions={2,3,4}, to_apply=add
      r3 = f32[1,2] reduce(add3, c0), dimensions={2,3,4}, to_apply=add
      r4 = f32[1,2] reduce(add4, c0), dimensions={2,3,4}, to_apply=add
      r5 = f32[1,2] reduce(add5, c0), dimensions={2,3,4}, to_apply=add

      out0 = f32[1,2] add(r1, r2)
      out1 = f32[1,2] add(r3, r4)
      out2 = f32[1,2] add(out0, out1)
      ROOT out3 = f32[1,2] add(out2, r5)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  ConstantDeferring pass;
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
      index("x.0"), index("y.0"), index("bcast1"), index("add1"),
      index("x.1"), index("y.1"), index("bcast2"), index("add2"),
      index("x.2"), index("y.2"), index("bcast3"), index("add3"),
      index("x.3"), index("y.3"), index("bcast4"), index("add4"),
      index("x.4"), index("y.4"), index("bcast5"), index("add5")};

  EXPECT_TRUE(absl::c_is_sorted(indices));
}

}  // namespace
}  // namespace xla
