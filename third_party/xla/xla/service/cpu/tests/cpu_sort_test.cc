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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

class CpuSortTest : public HloTestBase {};

// Regression test for https://github.com/openxla/xla/issues/47366: a sort
// comparator containing computation-calling ops (map, reduce) crashed the
// thunk emitter because the called computations were never emitted.
TEST_F(CpuSortTest, ComparatorCallingNestedComputations) {
  const std::string hlo_text = R"(
HloModule sort_comparator_with_nested_computations

add_f32 {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}

double_f32 {
  x = f32[] parameter(0)
  two = f32[] constant(2)
  ROOT mul = f32[] multiply(x, two)
}

compare {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  bcast0 = f32[2] broadcast(p0), dimensions={}
  bcast1 = f32[2] broadcast(p1), dimensions={}
  iota = f32[2] iota(), iota_dimension=0
  sum0 = f32[2] add(bcast0, iota)
  sum1 = f32[2] add(bcast1, iota)
  m0 = f32[2] map(sum0), to_apply=double_f32
  m1 = f32[2] map(sum1), to_apply=double_f32
  zero = f32[] constant(0)
  r0 = f32[] reduce(m0, zero), dimensions={0}, to_apply=add_f32
  r1 = f32[] reduce(m1, zero), dimensions={0}, to_apply=add_f32
  ROOT lt = pred[] compare(r0, r1), direction=LT
}

ENTRY entry {
  p = f32[8] parameter(0)
  ROOT sort = f32[8] sort(p), dimensions={0}, is_stable=true, to_apply=compare
}
)";

  // The comparator orders by 4*p+2, which is equivalent to p0 < p1.
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  Literal input = LiteralUtil::CreateR1<float>({3, -1, 7, 0, -5, 2, 8, 1});
  TF_ASSERT_OK_AND_ASSIGN(const Literal result,
                          Execute(std::move(module), {&input}));
  LiteralTestUtil::ExpectR1Equal<float>({-5, -1, 0, 1, 2, 3, 7, 8}, result);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
