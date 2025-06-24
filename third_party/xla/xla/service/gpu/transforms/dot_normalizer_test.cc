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

#include "xla/service/gpu/transforms/dot_normalizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using DotNormalizerTest = HloHardwareIndependentTestBase;
using ::tsl::testing::IsOkAndHolds;

TEST_F(DotNormalizerTest, DotWithoutContractingDims) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      p0 = f16[5,15]{1,0} parameter(0)
      p1 = f16[5,16,17]{2,1,0} parameter(1)
      ROOT r = f16[5,15,16,17]{3,2,1,0} dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DotNormalizer().Run(m.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      m->entry_computation()->root_instruction(),
      GmockMatch(
          m::Dot(m::Bitcast().WithShape(F16, {5, 15, 1}, {2, 1, 0}),
                 m::Bitcast().WithShape(F16, {5, 16, 17, 1}, {3, 2, 1, 0}))
              .WithContractingDims({2}, {3})));
}

TEST_F(DotNormalizerTest, DotWithContractingDims) {
  constexpr char kHlo[] = R"(
    HloModule test

    ENTRY main {
      p0 = f16[5,15,3]{2,1,0} parameter(0)
      p1 = f16[5,17,3]{2,1,0} parameter(1)
      ROOT r = f16[5,15,17]{2,1,0} dot(p0, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(DotNormalizer().Run(m.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
