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

// AOT compatibility of DeviceToHostCopyThunk and HostToDeviceCopyThunk (host
// memory offloading), on 1 GPU.

#include <memory>
#include <utility>

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

constexpr absl::string_view kTargetBase = "host_offload_copy_aot_test";

class HostOffloadCopyAotTest
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  HostOffloadCopyAotTest() : AotCompatibilityTest(GetParam()) {}
};

// HostOffloader turns the MoveToHost/MoveToDevice annotations into copies to
// and from host memory space S(5), and HostMemoryTransferAsyncifier turns those
// into copy-start/copy-done pairs, which lower to a D2H and an H2D copy thunk.
// Without the opt-barrier, the algebraic simplifier folds the two copies into
// a single device-to-device copy.
TEST_P(HostOffloadCopyAotTest, HostOffloadCopy) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      p = f32[4] parameter(0)
      to_host = f32[4] custom-call(p), custom_call_target="MoveToHost"
      tuple = (f32[4]) tuple(to_host)
      barrier = (f32[4]) opt-barrier(tuple)
      on_host = f32[4] get-tuple-element(barrier), index=0
      ROOT to_device = f32[4] custom-call(on_host),
                              custom_call_target="MoveToDevice"
    }
  )hlo";
  Literal p = LiteralUtil::CreateR1<float>({1, 2, 3, 4});
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> module,
      ParseAndReturnVerifiedModule(kHlo, GetModuleConfigForTest()));
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {&p}));
  LiteralTestUtil::ExpectR1Equal<float>({1, 2, 3, 4}, result);
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, HostOffloadCopyAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, HostOffloadCopyAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
