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

// AOT compatibility of ConvolutionThunk (cuDNN convolution), on 1 GPU. The
// cuDNN algorithm is pinned.

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/gpu/gpu_aot_test_params.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {
namespace {

using ::testing::ValuesIn;

constexpr absl::string_view kTargetBase = "convolution_aot_test";

class ConvolutionAotTest : public AotCompatibilityTest,
                           public ::testing::WithParamInterface<AotTestParam> {
 public:
  ConvolutionAotTest() : AotCompatibilityTest(GetParam()) {}
};

// The `xla_force_config` value: cuDNN engine 0 with its default knobs, on every
// arch. The heuristic's pick changes with the GPU and with the cuDNN version
// (on GB200, engine 28 at 9.23.0 and engine 0 at 9.25.1), and so would a golden
// left to it. Engine 0 needs no workspace, hence the empty `workspace_size`. If
// cuDNN no longer accepts the engine, the test fails: the runtime has no
// heuristic fallback yet. The config applies to every instruction that gets a
// config assigned, so the module must hold one convolution and no other such op
// (e.g. a dot).
constexpr absl::string_view kForcedCudnnConfig =
    "backend: CUDNN "
    "backend_config { algorithm { algo_id: 0 workspace_size {} } }";

// A sliding 2x2 window.
TEST_P(ConvolutionAotTest, Convolution) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule test
    ENTRY main {
      input = f32[1,1,3,3] parameter(0)
      filter = f32[1,1,2,2] parameter(1)
      ROOT conv = f32[1,1,2,2] convolution(input, filter), window={size=2x2},
                                       dim_labels=bf01_oi01->bf01
    }
  )hlo";
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions& debug_options = config.mutable_debug_options();
  debug_options.set_xla_gpu_autotune_level(0);
  debug_options.set_xla_force_config(kForcedCudnnConfig);
  // As a conv fusion the convolution would lower to CuDnnThunk, and the forced
  // algo_id would be read as a cuDNN plan index.
  debug_options.set_xla_gpu_experimental_enable_conv_fusion(false);
  Literal input =
      LiteralUtil::CreateR4<float>({{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}}});
  Literal filter = LiteralUtil::CreateR4<float>({{{{1, 2}, {3, 4}}}});
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo, config));
  ASSERT_OK_AND_ASSIGN(Literal result,
                       Execute(std::move(module), {&input, &filter}));
  // Cross-correlation (filter not flipped), e.g. 1*1 + 2*2 + 4*3 + 5*4 = 37.
  LiteralTestUtil::ExpectR4Near<float>({{{{37, 47}, {67, 77}}}}, result,
                                       ErrorSpec{1e-5, 1e-5});
}

INSTANTIATE_TEST_SUITE_P(BackwardsCompatibility, ConvolutionAotTest,
                         ValuesIn(GpuBackwardsCompatibilityParams(kTargetBase)),
                         AotTestParamName);
INSTANTIATE_TEST_SUITE_P(GoldenFileVerification, ConvolutionAotTest,
                         ValuesIn(GpuGoldenFileVerificationParams(kTargetBase)),
                         AotTestParamName);

}  // namespace
}  // namespace aot_compatibility
}  // namespace xla
