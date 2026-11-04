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

// AOT compatibility of TriangularSolveThunk inside a SequentialThunk (cuBLAS
// trsm), on 1 GPU.

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "triangular_solve_aot_test";

class TriangularSolveAotTest
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  TriangularSolveAotTest() : AotCompatibilityTest(GetParam()) {}
};

TEST_P(TriangularSolveAotTest, TriangularSolve) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test, entry_computation_layout={(f32[2,2]{0,1}, f32[2,2]{0,1})->f32[2,2]{0,1}}
    ENTRY main {
      a = f32[2,2]{0,1} parameter(0)
      b = f32[2,2]{0,1} parameter(1)
      ROOT x = f32[2,2]{0,1} triangular-solve(a, b), left_side=true, lower=true,
                                                     transpose_a=NO_TRANSPOSE
    }
  )hlo";
  Literal a = LiteralUtil::CreateR2<float>({{2, 0}, {1, 1}});
  Literal b = LiteralUtil::CreateR2<float>({{2, 4}, {3, 5}});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&a, &b}));
  // Row 0: 2 * x0 = {2, 4}; row 1: x0 + x1 = {3, 5}.
  LiteralTestUtil::ExpectR2Near<float>({{1, 2}, {2, 3}}, result,
                                       ErrorSpec{1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, TriangularSolveAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, TriangularSolveAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
