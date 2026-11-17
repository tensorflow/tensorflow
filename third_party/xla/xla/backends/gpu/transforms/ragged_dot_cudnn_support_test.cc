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

#include "xla/backends/gpu/transforms/ragged_dot_cudnn_support.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {
namespace {

class RaggedDotCuDnnSupportTest : public HloPjRtGpuTestBase {
 protected:
  se::StreamExecutor* stream_executor() const {
    auto platform =
        se::PlatformManager::PlatformWithId(stream_executor_platform_id());
    CHECK_OK(platform);
    absl::StatusOr<se::StreamExecutor*> executor =
        (*platform)->ExecutorForDevice(0);
    CHECK_OK(executor);
    return *executor;
  }

  Decision Probe(absl::string_view hlo) {
    absl::StatusOr<std::unique_ptr<HloModule>> module =
        ParseAndReturnVerifiedModule(hlo);
    CHECK_OK(module);
    module_ = *std::move(module);
    module_->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_use_ragged_dot_fusion(true);
    return IsRaggedDotSupportedByCuDnn(
        *module_->entry_computation()->root_instruction(), device_description(),
        stream_executor());
  }

  std::unique_ptr<HloModule> module_;
};

TEST_F(RaggedDotCuDnnSupportTest, SupportedRaggedDotIsAllowed) {
  if (!device_description().cuda_compute_capability().IsAtLeastAmpere()) {
    GTEST_SKIP() << "cuDNN ragged dot fusion requires Ampere or newer.";
  }
  if (device_description().dnn_version() < se::SemanticVersion{9, 22, 0}) {
    GTEST_SKIP() << "cuDNN ragged dot fusion requires cuDNN >= 9.22.";
  }
  Decision decision = Probe(R"(
    HloModule test

    ENTRY main {
      p0 = bf16[128,512]{1,0} parameter(0)
      p1 = bf16[16,512,256]{2,1,0} parameter(1)
      p2 = s32[16]{0} parameter(2)
      ROOT rd = bf16[128,256]{1,0} ragged-dot(p0, p1, p2),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
    })");
  EXPECT_TRUE(decision.IsAllowed()) << decision.Explain();
}

// cuDNN has no engine for mixed token/weight dtypes, so the probe must report
// a forbidden decision (the caller then expands to a dense dot).
TEST_F(RaggedDotCuDnnSupportTest, UnsupportedRaggedDotIsForbidden) {
  Decision decision = Probe(R"(
    HloModule test

    ENTRY main {
      p0 = bf16[128,512]{1,0} parameter(0)
      p1 = f16[16,512,256]{2,1,0} parameter(1)
      p2 = s32[16]{0} parameter(2)
      ROOT rd = bf16[128,256]{1,0} ragged-dot(p0, p1, p2),
          lhs_contracting_dims={1}, rhs_contracting_dims={1},
          lhs_ragged_dims={0}, rhs_group_dims={0}
    })");
  EXPECT_TRUE(decision.IsForbidden());
  EXPECT_FALSE(decision.Explain().empty());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
