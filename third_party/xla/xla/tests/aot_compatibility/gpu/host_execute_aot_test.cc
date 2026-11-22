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

// AOT compatibility of HostExecuteStartThunk and HostExecuteDoneThunk (host
// compute offloading), on 1 GPU.
//
// The module is a `call` with the `_xla_compute_type="host"` frontend
// attribute, as jax.experimental.compute_on("device_host") emits it; the GPU
// compiler rewrites it into the HostExecute start/done pair. The embedded
// XLA:CPU code is compiled for the host that dumped the golden.

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

constexpr absl::string_view kTargetBase = "host_execute_aot_test";

class HostExecuteAotTest : public AotCompatibilityTest,
                           public ::testing::WithParamInterface<AotTestParam> {
 public:
  HostExecuteAotTest() : AotCompatibilityTest(GetParam()) {}
};

// The tuple result gives the start and done thunks two result buffers each.
// The results differ from each other, and subtract and divide are not
// commutative, so the values also check buffer and argument order.
TEST_P(HostExecuteAotTest, HostExecuteTupleResult) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    host_fn {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      sub = f32[4] subtract(p0, p1)
      div = f32[4] divide(p0, p1)
      ROOT tuple = (f32[4], f32[4]) tuple(sub, div)
    }
    ENTRY main {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT call = (f32[4], f32[4]) call(p0, p1), to_apply=host_fn,
          frontend_attributes={_xla_compute_type="host"}
    }
  )hlo";
  Literal p0 = LiteralUtil::CreateR1<float>({8, 27, 64, 125});
  Literal p1 = LiteralUtil::CreateR1<float>({2, 3, 4, 5});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&p0, &p1}));
  std::vector<Literal> elements = result.DecomposeTuple();
  ASSERT_EQ(elements.size(), 2);
  LiteralTestUtil::ExpectR1Equal<float>({6, 24, 60, 120}, elements[0]);
  LiteralTestUtil::ExpectR1Equal<float>({4, 9, 16, 25}, elements[1]);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, HostExecuteAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, HostExecuteAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
