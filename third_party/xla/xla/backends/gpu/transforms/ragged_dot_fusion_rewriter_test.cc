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

#include "xla/backends/gpu/transforms/ragged_dot_fusion_rewriter.h"

#include <array>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

using ::testing::HasSubstr;
using ::testing::Not;

static const std::initializer_list<absl::string_view> kbf16f16{"bf16", "f16"};

// This class performs isolated unit testing of the RaggedDotFusionRewriter
// pass. It verifies that specific HLO patterns are correctly recognized and
// rewritten into cudnn fusions.
class RaggedDotFusionRewriterUnitTest : public HloPjRtGpuTestBase {
 public:
  bool IsCuda() const {
    return device_description().gpu_compute_capability().IsCuda();
  }
  se::CudaComputeCapability GetCudaComputeCapability() const {
    return device_description().cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() const {
    auto version = device_description().dnn_version();
    return stream_executor::dnn::VersionInfo(version.major(), version.minor(),
                                             version.patch());
  }

  se::SemanticVersion GetToolkitVersion() const {
    return device_description().runtime_version();
  }

  RaggedDotFusionRewriter GetRaggedDotFusionRewriter() const {
    return RaggedDotFusionRewriter();
  }

  template <typename Pattern>
  void RunAndMatch(absl::string_view hlo_string, Pattern&& fusion_matcher) {
    TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));

    RaggedDotFusionRewriter rewriter = GetRaggedDotFusionRewriter();
    TF_ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

    SCOPED_TRACE(m->ToString());
    EXPECT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(std::forward<Pattern>(fusion_matcher)));
  }

  RaggedDotFusionRewriterUnitTest()
      : HloPjRtGpuTestBase(HloPjRtTestBaseOptions{
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/false,
            /*instruction_can_change_layout_func=*/{}}) {}
};

TEST_F(RaggedDotFusionRewriterUnitTest, TestSupportedRaggedDot) {
  RunAndMatch(R"(
    HloModule Test

    ENTRY Test {
      input = bf16[128,512]{1,0} parameter(0)
      weight = bf16[16,512,256]{2,1,0} parameter(1)
      group_sizes = s32[16]{0} parameter(2)
      ROOT rd = bf16[128,256]{1,0} ragged-dot(input, weight, group_sizes),
             lhs_contracting_dims={1}, rhs_contracting_dims={1}, lhs_ragged_dims={0}, rhs_group_dims={0}
    })",
              m::Fusion()
                  .WithFusionKind(HloInstruction::FusionKind::kCustom)
                  .WithShape(BF16, {128, 256}));
}

// This class performs end-to-end integration testing of the RaggedDotRewriter.
// It verifies that the rewriter works correctly within the full GPU
// optimization pipeline and produces numerically correct results on hardware.
class RaggedDotFusionRewriterIntegrationTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtGpuTestBase>,
      public ::testing::WithParamInterface<absl::string_view> {
 public:
  bool IsCuda() const {
    return device_description().gpu_compute_capability().IsCuda();
  }
  se::CudaComputeCapability GetCudaComputeCapability() const {
    return device_description().cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() const {
    auto version = device_description().dnn_version();
    return stream_executor::dnn::VersionInfo(version.major(), version.minor(),
                                             version.patch());
  }

  stream_executor::SemanticVersion GetToolkitVersion() const {
    return device_description().runtime_version();
  }

  RaggedDotFusionRewriterIntegrationTest()
      : HloPjRtInterpreterReferenceMixin<HloPjRtGpuTestBase>(
            HloPjRtTestBaseOptions{
                /*verifier_layout_sensitive=*/false,
                /*allow_mixed_precision_in_hlo_verifier=*/false,
                /*instruction_can_change_layout_func=*/{}}) {}

 protected:
  std::string GetOptimizedHlo(absl::string_view hlo_string) {
    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions debug_opts = config.debug_options();
    debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
    config.set_debug_options(debug_opts);

    absl::StatusOr<std::unique_ptr<HloModule>> module_or_status =
        GetOptimizedModule(hlo_string, config);
    if (!module_or_status.ok()) {
      TF_EXPECT_OK(module_or_status.status());
      return "";
    }
    std::unique_ptr<HloModule> module = std::move(module_or_status.value());
    HloPrintOptions print_opts;
    print_opts.set_print_operand_shape(false);
    return module->ToString(print_opts);
  }
};

TEST_P(RaggedDotFusionRewriterIntegrationTest, TestRaggedDotOnly) {
  if (GetDnnVersion() < se::dnn::VersionInfo{9, 21, 0}) {
    GTEST_SKIP() << "CuDNN ragged dot requires cuDNN 9.21+.";
  }

  const std::string hlo_with_new_type =
      absl::StrReplaceAll(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[128,512]{1,0} parameter(0)
      weight = TYPE[16,512,256]{2,1,0} parameter(1)
      group_sizes = s32[16]{0} parameter(2)
      ROOT rd = TYPE[128,256]{1,0} ragged-dot(input, weight, group_sizes),
             lhs_contracting_dims={1}, rhs_contracting_dims={1}, lhs_ragged_dims={0}, rhs_group_dims={0}
    })",
                          {{"TYPE", GetParam()}});
  std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
  EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_with_new_type));
  DebugOptions debug_opts = module->config().debug_options();
  debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  module->mutable_config().set_debug_options(debug_opts);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}))
      << optimized_hlo_string;
}

INSTANTIATE_TEST_SUITE_P(AllTypes, RaggedDotFusionRewriterIntegrationTest,
                         ::testing::ValuesIn(kbf16f16));
}  // namespace
}  // namespace gpu
}  // namespace xla
