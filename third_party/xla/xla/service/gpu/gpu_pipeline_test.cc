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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/core/host_offloading/hlo_host_device_type_call_wrapper.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pass_options.h"
#include "xla/hlo/pass/hlo_pass_registry.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

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
  // Every set field becomes an option that the algebraic simplifier must
  // consume.
  AlgebraicSimplifierOptions algsimp_opts;
  algsimp_opts.set_unconditionally_simplify_reduce_of_transpose_or_reshape(
      true);
  algsimp_opts.set_minmax_propagate_nan(false);
  const HloPipelineProto proto = GetDefaultCompilationPipeline(algsimp_opts);
  std::vector<const HloPipelineNodeProto*> passes;
  CollectPasses(proto.root(), passes);
  ASSERT_THAT(passes, Not(IsEmpty()));

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
  for (const HloPipelineNodeProto* pass : passes) {
    HloPassOptions options(pass->options());
    EXPECT_OK(HloPassRegistry::Global().Create(pass->name(), context, options))
        << pass->name();
  }
}

}  // namespace
}  // namespace xla::gpu
