/* Copyright 2017 The OpenXLA Authors.

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

// This demonstrates how to use hlo_test_base to create textual IR based
// testcases.

#include <optional>
#include <string>

#include "xla/hlo/testlib/test.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla {
namespace {

using std::nullopt;

using ConstantReductionFunctionTest =
    HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(ConstantReductionFunctionTest, Bool) {
  const std::string& hlo_string = R"(
HloModule jit_f__2.10

reduction_computation__3.4 {
  parameter.5 = pred[] parameter(0)
  parameter.6 = pred[] parameter(1)
  constant.7 = pred[] constant(false)
  ROOT constant.8 = pred[] constant(true)
}

ENTRY jit_f__2.10 {
  constant.2 = pred[] constant(false)
  parameter.1 = pred[24,1,1,5]{3,2,1,0} parameter(0)
  constant.3 = pred[] constant(false)
  ROOT reduce.9 = pred[24,1,1]{2,1,0} reduce(parameter.1, constant.3), dimensions={3}, to_apply=reduction_computation__3.4
}
)";
  EXPECT_TRUE(RunAndCompare(hlo_string, nullopt));
}

}  // namespace
}  // namespace xla
