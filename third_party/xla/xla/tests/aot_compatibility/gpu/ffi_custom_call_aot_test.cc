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

// AOT compatibility of a typed-FFI CustomCallThunk (RAFT TopK `__gpu$TopK`),
// on 1 GPU.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
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

constexpr absl::string_view kTargetBase = "ffi_custom_call_aot_test";

class FfiCustomCallAotTest
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  FfiCustomCallAotTest() : AotCompatibilityTest(GetParam()) {}
};

TEST_P(FfiCustomCallAotTest, CustomCall) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    compare {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT gt = pred[] compare(lhs, rhs), direction=GT
    }
    ENTRY main {
      p = f32[8] parameter(0)
      ROOT topk = (f32[1], s32[1]) custom-call(p), custom_call_target="TopK",
                  to_apply=compare, backend_config="{is_stable = false}"
    }
  )hlo";
  Literal p = LiteralUtil::CreateR1<float>({3, 1, 4, 1, 5, 9, 2, 6});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&p}));
  std::vector<Literal> outputs = result.DecomposeTuple();
  LiteralTestUtil::ExpectR1Equal<float>({9}, outputs[0]);
  LiteralTestUtil::ExpectR1Equal<int32_t>({5}, outputs[1]);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, FfiCustomCallAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, FfiCustomCallAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
