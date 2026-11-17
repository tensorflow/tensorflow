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

#include "xla/service/gpu/gpu_pipeline.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;

void CollectPasses(const HloPipelineNodeSequenceProto& sequence,
                   std::vector<const HloPipelineNodeProto*>& passes) {
  for (const HloPipelineNodeProto& node : sequence.nodes()) {
    if (node.has_sub_nodes()) {
      CollectPasses(node.sub_nodes(), passes);
    } else {
      passes.push_back(&node);
    }
  }
}

// Checks that every pass of the default pipeline is registered and consumes all
// of its options.
TEST(GpuPipelineTest, DefaultCompilationPipelinePassesAreCreatable) {
  const se::DeviceDescription device_description =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  // Creatability doesn't depend on what the backend hooks do.
  const auto noop = [](HloInstruction*) { return absl::OkStatus(); };
  const HloHostDeviceTypeCallWrapper::Options host_device_type_options{
      /*set_backend_config_fn=*/noop,
      /*clear_backend_config_device_type=*/noop,
  };
  HloPassContext context;
  ASSERT_OK(context.Set(device_description));
  ASSERT_OK(context.Set(host_device_type_options));

  ASSERT_OK_AND_ASSIGN(
      const HloPipelineProto proto,
      GetDefaultCompilationPipeline(HloModuleConfig(), context));
  std::vector<const HloPipelineNodeProto*> passes;
  CollectPasses(proto.root(), passes);
  ASSERT_THAT(passes, Not(IsEmpty()));

  for (const HloPipelineNodeProto* pass : passes) {
    HloPassOptions options(pass->options());
    EXPECT_OK(HloPassRegistry::Global().Create(pass->name(), context, options))
        << pass->name();
  }
}

TEST(GpuPipelineTest, MissingDeviceDescriptionInContextFails) {
  EXPECT_THAT(
      GetDefaultCompilationPipeline(HloModuleConfig(), HloPassContext()),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("DeviceDescription")));
}

using GpuPipelineAlgebraicSimplifierTest = HloHardwareIndependentTestBase;

TEST_F(GpuPipelineAlgebraicSimplifierTest,
       ConvolutionCanonicalizationRespectsFastMinMax) {
  constexpr absl::string_view kHloText = R"hlo(
    HloModule minmax

    ENTRY main {
      p = f32[] parameter(0)
      negative_infinity = f32[] constant(-inf)
      positive_infinity = f32[] constant(inf)
      maximum = f32[] maximum(p, negative_infinity)
      minimum = f32[] minimum(p, positive_infinity)
      ROOT result = (f32[], f32[]) tuple(maximum, minimum)
    })hlo";
  for (bool fast_min_max : {false, true}) {
    SCOPED_TRACE(fast_min_max);
    for (bool is_cuda_platform : {false, true}) {
      SCOPED_TRACE(is_cuda_platform);
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                           ParseAndReturnVerifiedModule(kHloText));
      DebugOptions debug_options = GetDebugOptionsForTest();
      debug_options.set_xla_gpu_enable_fast_min_max(fast_min_max);
      AlgebraicSimplifier simplifier(GetAlgebraicSimplifierOptions(
          AlgebraicSimplifierMode::kGpuConvoluationCanonicalization,
          debug_options, is_cuda_platform));
      EXPECT_THAT(simplifier.Run(module.get()), IsOkAndHolds(!fast_min_max));
      const HloInstruction* root =
          module->entry_computation()->root_instruction();
      if (fast_min_max) {
        EXPECT_THAT(root, GmockMatch(m::Tuple(
                              m::Maximum(m::Parameter(0), m::Constant()),
                              m::Minimum(m::Parameter(0), m::Constant()))));
      } else {
        EXPECT_THAT(root,
                    GmockMatch(m::Tuple(m::Parameter(0), m::Parameter(0))));
      }
    }
  }
}

}  // namespace
}  // namespace xla::gpu
