/* Copyright 2018 Graphcore. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/hlo_computation_name_uniquify.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include "absl/algorithm/container.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloComputationNameUniquifyTest = HloTestBase;

TEST_F(HloComputationNameUniquifyTest, RenameResarved) {
  std::string hlo = R"(
HloModule top

_pop_op_test {
  ROOT arg = f16[4] parameter(0)
}

__arithmetic_test {
  ROOT arg = f16[4] parameter(0)
}

__inline_test {
  ROOT arg = f16[4] parameter(0)
}

__repeat_test {
  ROOT arg = f16[4] parameter(0)
}

ENTRY _pop_op_test2 {
  p0 = f16[4] parameter(0)
  p1 = f16[4] parameter(1)
  p2 = f16[4] parameter(2)
  p3 = f16[4] parameter(3)

  out0 = f16[4] call(p0), to_apply=_pop_op_test
  out1 = f16[4] call(p1), to_apply=__arithmetic_test
  out2 = f16[4] call(p2), to_apply=__inline_test
  out3 = f16[4] call(p3), to_apply=__repeat_test

  ROOT t = (f16[4], f16[4], f16[4], f16[4]) tuple(out0, out1, out2, out3)
}
)";

  auto module = ParseHloString(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  HloComputationNameUniquify hcnu;
  EXPECT_TRUE(hcnu.Run(module0).ValueOrDie());

  // Expect all computations begin with 'a'
  auto comp_name_starts_with_a = [&](HloComputation* comp) {
    return tensorflow::str_util::StartsWith(comp->name(), "a");
  };
  EXPECT_TRUE(absl::c_all_of(module0->computations(), comp_name_starts_with_a));
}

TEST_F(HloComputationNameUniquifyTest, DontRename) {
  std::string hlo = R"(
HloModule top

__pop_op_test {
  ROOT arg = f16[4] parameter(0)
}

__aithmetic_test {
  ROOT arg = f16[4] parameter(0)
}

__outline_test {
  ROOT arg = f16[4] parameter(0)
}

_a_repeat_test {
  ROOT arg = f16[4] parameter(0)
}

ENTRY b_pop_op_test2 {
  p0 = f16[4] parameter(0)
  p1 = f16[4] parameter(1)
  p2 = f16[4] parameter(2)
  p3 = f16[4] parameter(3)

  out0 = f16[4] call(p0), to_apply=__pop_op_test
  out1 = f16[4] call(p1), to_apply=__aithmetic_test
  out2 = f16[4] call(p2), to_apply=__outline_test
  out3 = f16[4] call(p3), to_apply=_a_repeat_test

  ROOT t = (f16[4], f16[4], f16[4], f16[4]) tuple(out0, out1, out2, out3)
}
)";

  auto module = ParseHloString(hlo, GetModuleConfigForTest());
  EXPECT_TRUE(module.ok());
  auto* module0 = module.ValueOrDie().get();

  HloComputationNameUniquify hcnu;
  EXPECT_FALSE(hcnu.Run(module0).ValueOrDie());
  // Expect none of computations begin with 'a'
  auto comp_name_starts_with_a = [&](HloComputation* comp) {
    return tensorflow::str_util::StartsWith(comp->name(), "a");
  };
  EXPECT_TRUE(
      absl::c_none_of(module0->computations(), comp_name_starts_with_a));
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
