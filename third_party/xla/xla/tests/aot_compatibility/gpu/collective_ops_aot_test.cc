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

// LEGACY: do not add test cases here. New collective coverage goes into the
// per-thunk-family targets described in ../google/README.md.
//
// These are the original collective AOT smoke tests. They are kept only for
// their goldens from older compilers (v1-v3), the oldest golden history in the
// tree. The family targets cover the same HLO in their ElementTypes cases.
//
// TODO(aliia): Delete this target, its goldens and its README entry once
// reduction_collectives_aot_test and data_movement_collectives_aot_test have
// at least two golden versions on every architecture. From then on, their own
// backwards-compatibility suites replay goldens from an older compiler.

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/literal.h"
#include "xla/tests/aot_compatibility/gpu/collective_aot_test_base.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "collective_ops_aot_test_2gpu";
constexpr int64_t kNumReplicas = 2;

class CollectiveOpsAotTest : public CollectiveAotTestBase {};

TEST_P(CollectiveOpsAotTest, AllGatherMixedTypes) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY test_computation {
      id = u32[] replica-id()
      p0 = u32[2, 1] broadcast(id), dimensions={}
      p1 = f32[2, 1] convert(p0)
      allgather = (u32[2, 2], f32[2, 2]) all-gather(p0, p1), dimensions={1}
      ag0 = u32[2, 2] get-tuple-element(allgather), index=0
      ag1 = f32[2, 2] get-tuple-element(allgather), index=1
      r0 = u32[4] reshape(ag0)
      r1 = f32[4] reshape(ag1)
      ROOT out = (u32[4], f32[4]) tuple(r0, r1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      RunReplicated(kHlo, GetModuleConfigForTest(kNumReplicas)));
  for (int64_t replica = 0; replica < kNumReplicas; ++replica) {
    std::vector<Literal> outputs = results[replica].DecomposeTuple();
    LiteralTestUtil::ExpectR1Equal<uint32_t>({0, 1, 0, 1}, outputs[0]);
    LiteralTestUtil::ExpectR1Near<float>({0.0, 1.0, 0.0, 1.0}, outputs[1],
                                         ErrorSpec{1e-5, 1e-5});
  }
}

TEST_P(CollectiveOpsAotTest, ReduceScatter) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    add {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT add = u32[] add(lhs, rhs)
    }

    ENTRY main {
      c0 = u32[8] constant({1, 2, 3, 4, 5, 6, 7, 8})
      c1 = u32[8] constant({10, 11, 12, 13, 14, 15, 16, 17})
      zero = u32[] constant(0)
      id = u32[] replica-id()
      p = pred[] compare(id, zero), direction=EQ
      pb = pred[8] broadcast(p), dimensions={}
      // data = c0 for replica 0 and c1 for replica 1
      data = u32[8] select(pb, c0, c1)
      ROOT ars = u32[4] reduce-scatter(data), replica_groups={},
                        dimensions={0}, to_apply=add
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      RunReplicated(kHlo, GetModuleConfigForTest(kNumReplicas)));
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 13, 15, 17}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({19, 21, 23, 25}, results[1]);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, CollectiveOpsAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, CollectiveOpsAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
