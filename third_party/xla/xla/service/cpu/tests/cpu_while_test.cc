/* Copyright 2022 The OpenXLA Authors.

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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/tests/literal_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

// Verifies fix for b/233647273.
TEST_F(CpuCodegenTest, While) {
  const std::string hlo_text = R"(
HloModule module

f1 {
  f1.p0 = s32[] parameter(0)
  ROOT f1.sum = s32[] add(f1.p0, f1.p0)
}

f2 {
  f2.p0 = s32[] parameter(0)
  f2.p1 = s32[] parameter(1)
  ROOT f2.sum = s32[] add(f2.p0, f2.p1)
}

body {
  body.p0 = s32[] parameter(0)
  sum2 = s32[] fusion(body.p0), kind=kLoop, calls=f1
  ROOT sum3 = s32[] fusion(sum2, body.p0), kind=kLoop, calls=f2
}

cond {
  cond.p0 = s32[] parameter(0)
  cond.c1 = s32[] constant(1)
  ROOT cond.root = pred[] compare(cond.p0, cond.c1), direction=EQ
}

ENTRY entry {
  entry.c1 = s32[] constant(1)
  ROOT entry.root = s32[] while(entry.c1), condition=cond, body=body
}
)";

  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));

  // Compile and execute the computation.
  auto result = ExecuteAndTransfer(module->Clone(), {});

  // Check the output correctness.
  LiteralTestUtil::ExpectR0Equal(3, result);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
