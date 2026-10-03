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

#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
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
#include "xla/hlo/transforms/expanders/ragged_dot_rewriter.h"
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
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

namespace m = match;

using ::testing::HasSubstr;
using ::testing::Not;

static const std::initializer_list<absl::string_view> kbf16f16{"bf16", "f16"};
static const std::initializer_list<absl::string_view> ks32s64{"s32", "s64"};

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
    return stream_executor::dnn::VersionInfo(version.major_version(),
                                             version.minor_version(),
                                             version.patch_version());
  }

  se::SemanticVersion GetToolkitVersion() const {
    return device_description().runtime_version();
  }

  RaggedDotFusionRewriter GetRaggedDotFusionRewriter() const {
    return RaggedDotFusionRewriter();
  }

  template <typename Pattern>
  void RunAndMatch(absl::string_view hlo_string, Pattern&& fusion_matcher) {
    ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(hlo_string));

    RaggedDotFusionRewriter rewriter = GetRaggedDotFusionRewriter();
    ASSERT_OK(RunHloPass(&rewriter, m.get()).status());

    SCOPED_TRACE(m->ToString());
    EXPECT_THAT(m->entry_computation()->root_instruction(),
                GmockMatch(std::forward<Pattern>(fusion_matcher)));
  }

  RaggedDotFusionRewriterUnitTest()
      : HloPjRtGpuTestBase(
            HloTestBaseOptions{/*verifier_layout_sensitive=*/false,
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

// Wgrad: contracts over the ragged M dimension, producing dweight [G, K, N].
// Corresponds to cuDNN moe_grouped_matmul_bwd.
TEST_F(RaggedDotFusionRewriterUnitTest, TestSupportedRaggedDotWgrad) {
  RunAndMatch(R"(
    HloModule Test

    ENTRY Test {
      input = bf16[128,512]{1,0} parameter(0)
      doutput = bf16[128,256]{1,0} parameter(1)
      group_sizes = s32[16]{0} parameter(2)
      ROOT rd = bf16[16,512,256]{2,1,0} ragged-dot(input, doutput, group_sizes),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
    })",
              m::Fusion()
                  .WithFusionKind(HloInstruction::FusionKind::kCustom)
                  .WithShape(BF16, {16, 512, 256}));
}

// cuDNN's `FirstTokenOffset` only encodes each group's *start* offset (see
// MaxGroupedMatmul docs: "the total token count [is] implicit from the token
// tensor dimension"), so the *last* group's true end can never be recovered
// from the offset array alone -- cuDNN always implicitly extends it to the
// token tensor's full static row count instead. For the wgrad flavor (which
// actually reduces over the ragged/token dimension into a real, consumed
// output) that means whatever happens to occupy the unused tail rows of the
// ragged buffer gets summed into the last group's weight gradient unless XLA
// explicitly masks it first. Verify the rewriter inserts that masking
// (select/iota/compare, see MaskRaggedDotPaddingTail) ahead of the fusion for
// both ragged-dot operands, rather than feeding them into the fusion
// unmasked.
TEST_F(RaggedDotFusionRewriterUnitTest,
       TestRaggedDotWgradMasksPaddingTailBeforeLastGroupIsAmbiguous) {
  RunAndMatch(R"(
    HloModule Test

    ENTRY Test {
      input = bf16[128,512]{1,0} parameter(0)
      doutput = bf16[128,256]{1,0} parameter(1)
      group_sizes = s32[16]{0} parameter(2)
      ROOT rd = bf16[16,512,256]{2,1,0} ragged-dot(input, doutput, group_sizes),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
    })",
              m::Fusion(m::Select(), m::Select(), m::Subtract())
                  .WithFusionKind(HloInstruction::FusionKind::kCustom)
                  .WithShape(BF16, {16, 512, 256}));
}

// The non-wgrad (dInput/forward) flavor doesn't need the padding-tail mask:
// its ragged dimension only selects which *output* rows are valid, so
// garbage in the unused tail just lands in output rows that are never
// consumed downstream. Verify the rewriter does NOT insert masking there --
// the fusion's operands should be the raw input/weight/offset values
// unchanged, matching TestSupportedRaggedDot's shape/kind-only check but
// additionally pinning down the operand chain so a regression that started
// masking this (harmless but wasteful) case would be caught.
TEST_F(RaggedDotFusionRewriterUnitTest,
       TestSupportedRaggedDotDoesNotMaskPaddingTail) {
  RunAndMatch(R"(
    HloModule Test

    ENTRY Test {
      input = bf16[128,512]{1,0} parameter(0)
      weight = bf16[16,512,256]{2,1,0} parameter(1)
      group_sizes = s32[16]{0} parameter(2)
      ROOT rd = bf16[128,256]{1,0} ragged-dot(input, weight, group_sizes),
             lhs_contracting_dims={1}, rhs_contracting_dims={1}, lhs_ragged_dims={0}, rhs_group_dims={0}
    })",
              m::Fusion(m::Parameter(), m::Parameter(), m::Subtract())
                  .WithFusionKind(HloInstruction::FusionKind::kCustom)
                  .WithShape(BF16, {128, 256}));
}

// This class performs end-to-end integration testing of the RaggedDotRewriter.
// It verifies that the rewriter works correctly within the full GPU
// optimization pipeline and produces numerically correct results on hardware.
class RaggedDotFusionRewriterIntegrationTest
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase>,
      public ::testing::WithParamInterface<
          std::tuple<absl::string_view, absl::string_view>> {
 public:
  bool IsCuda() const {
    return device_description().gpu_compute_capability().IsCuda();
  }
  se::CudaComputeCapability GetCudaComputeCapability() const {
    return device_description().cuda_compute_capability();
  }
  stream_executor::dnn::VersionInfo GetDnnVersion() const {
    auto version = device_description().dnn_version();
    return stream_executor::dnn::VersionInfo(version.major_version(),
                                             version.minor_version(),
                                             version.patch_version());
  }

  stream_executor::SemanticVersion GetToolkitVersion() const {
    return device_description().runtime_version();
  }

  // Same runtime cuDNN version check RaggedDotRewriter uses to decide
  // whether to route the ragged dot wgrad through the cuDNN fusion path.
  bool SupportsCudnnRaggedDotWgrad() const {
    return GetDnnVersion() >= kMinCudnnVersionForRaggedDotWgradFusion;
  }

  RaggedDotFusionRewriterIntegrationTest()
      : HloInterpreterReferenceMixin<HloPjRtGpuTestBase>(
            HloTestBaseOptions{/*verifier_layout_sensitive=*/false,
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
      EXPECT_OK(module_or_status.status());
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

  const auto& [data_type, group_type] = GetParam();
  const std::string hlo_with_new_type =
      absl::StrReplaceAll(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[128,512]{1,0} parameter(0)
      weight = TYPE[16,512,256]{2,1,0} parameter(1)
      group_sizes = GROUP_TYPE[16]{0} constant({7,9,6,10,8,8,8,8,8,8,8,8,8,8,8,8})
      ROOT rd = TYPE[128,256]{1,0} ragged-dot(input, weight, group_sizes),
             lhs_contracting_dims={1}, rhs_contracting_dims={1}, lhs_ragged_dims={0}, rhs_group_dims={0}
    })",
                          {{"TYPE", data_type}, {"GROUP_TYPE", group_type}});
  std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
  EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_with_new_type));
  DebugOptions debug_opts = module->config().debug_options();
  debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  module->mutable_config().set_debug_options(debug_opts);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}))
      << optimized_hlo_string;
}

// Wgrad: contracts over the ragged M dimension (kRaggedContracting).
// Uses cuDNN moe_grouped_matmul_bwd to compute dweight[G,K,N].
TEST_P(RaggedDotFusionRewriterIntegrationTest, TestRaggedDotWgrad) {
  if (!SupportsCudnnRaggedDotWgrad()) {
    GTEST_SKIP() << "CuDNN ragged dot wgrad requires cuDNN 9.24+.";
  }

  const auto& [data_type, group_type] = GetParam();
  const std::string hlo_with_new_type =
      absl::StrReplaceAll(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[128,512]{1,0} parameter(0)
      doutput = TYPE[128,256]{1,0} parameter(1)
      group_sizes = GROUP_TYPE[16]{0} constant({7,9,6,10,8,8,8,8,8,8,8,8,8,8,8,8})
      ROOT rd = TYPE[16,512,256]{2,1,0} ragged-dot(input, doutput, group_sizes),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
    })",
                          {{"TYPE", data_type}, {"GROUP_TYPE", group_type}});
  std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
  EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_with_new_type));
  DebugOptions debug_opts = module->config().debug_options();
  debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  module->mutable_config().set_debug_options(debug_opts);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}))
      << optimized_hlo_string;
}

// Wgrad where `group_sizes` doesn't sum to the full ragged (M) dimension,
// leaving the last group empty and rows [120, 128) of `input`/`doutput` as
// pure padding. cuDNN's `FirstTokenOffset` only encodes each group's *start*
// offset (group 15 starts at row 120), so its true end (also row 120, since
// its size is 0) can't be recovered from the offset array alone -- cuDNN
// implicitly extends the last group all the way to the token tensor's full
// static row count (128) instead. If XLA didn't explicitly zero that unused
// tail before handing operands to cuDNN (see MaskRaggedDotPaddingTail), the
// last group's weight gradient would incorporate whatever data happens to
// occupy those padding rows -- here, ordinary (non-zero) random test data --
// producing a non-zero dweight[15,:,:] instead of the correct all-zero
// result, and this test would fail against the interpreter reference.
TEST_P(RaggedDotFusionRewriterIntegrationTest,
       TestRaggedDotWgradLastGroupSizeNotDerivableFromOffset) {
  if (!SupportsCudnnRaggedDotWgrad()) {
    GTEST_SKIP() << "CuDNN ragged dot wgrad requires cuDNN 9.24+.";
  }

  const auto& [data_type, group_type] = GetParam();
  const std::string hlo_with_new_type =
      absl::StrReplaceAll(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[128,512]{1,0} parameter(0)
      doutput = TYPE[128,256]{1,0} parameter(1)
      group_sizes = GROUP_TYPE[16]{0} constant({7,9,6,10,8,8,8,8,8,8,8,8,8,8,8,0})
      ROOT rd = TYPE[16,512,256]{2,1,0} ragged-dot(input, doutput, group_sizes),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
    })",
                          {{"TYPE", data_type}, {"GROUP_TYPE", group_type}});
  std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
  EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_with_new_type));
  DebugOptions debug_opts = module->config().debug_options();
  debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  module->mutable_config().set_debug_options(debug_opts);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}))
      << optimized_hlo_string;
}

// Wgrad with K and N sizes that are not 16-byte aligned (not a multiple of 8
// elements for bf16/f16). cuDNN's wgrad path lowers to a cuBLASLt grouped
// GEMM, which relies on TMA and requires 16B alignment on the contiguous
// (K/N) dimensions of each matrix; since K and N are static shapes, XLA is
// expected to pad them at compile time rather than fail or require runtime
// padding based on group_sizes.
TEST_P(RaggedDotFusionRewriterIntegrationTest, TestRaggedDotWgradUnalignedKN) {
  if (!SupportsCudnnRaggedDotWgrad()) {
    GTEST_SKIP() << "CuDNN ragged dot wgrad requires cuDNN 9.24+.";
  }

  const auto& [data_type, group_type] = GetParam();
  const std::string hlo_with_new_type =
      absl::StrReplaceAll(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[128,510]{1,0} parameter(0)
      doutput = TYPE[128,254]{1,0} parameter(1)
      group_sizes = GROUP_TYPE[16]{0} constant({8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8})
      ROOT rd = TYPE[16,510,254]{2,1,0} ragged-dot(input, doutput, group_sizes),
             lhs_contracting_dims={0}, rhs_contracting_dims={0}, lhs_ragged_dims={0}
    })",
                          {{"TYPE", data_type}, {"GROUP_TYPE", group_type}});
  std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
  EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));
  // K (510) and N (254) are not 16-byte aligned for bf16/f16 (need a
  // multiple of 8 elements). Verify XLA actually pads them at compile time
  // to 512/256 rather than silently skipping the padding -- RunAndCompare
  // below would still pass numerically even if the padding step were
  // skipped and cuDNN just tolerated the misalignment, so that alone isn't
  // enough to catch a regression here.
  EXPECT_THAT(optimized_hlo_string, HasSubstr("pad("));
  EXPECT_THAT(optimized_hlo_string, HasSubstr("512"));
  EXPECT_THAT(optimized_hlo_string, HasSubstr("256"));

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_with_new_type));
  DebugOptions debug_opts = module->config().debug_options();
  debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  module->mutable_config().set_debug_options(debug_opts);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}))
      << optimized_hlo_string;
}

// Wgrad where M (the ragged/contracting dimension) is the leading
// (fastest-moving/minor) dimension of the lhs/rhs operands instead of K/N,
// e.g. input[K,M] instead of input[M,K]. Since M's per-group sizes are only
// known at runtime, XLA is expected to transpose such operands so that K/N
// becomes the leading dimension instead, and then pad K/N (which are static
// and thus safe to pad at compile time) up to the required alignment. K and
// N are also chosen to be unaligned here to exercise both the transpose and
// the padding together.
TEST_P(RaggedDotFusionRewriterIntegrationTest,
       TestRaggedDotWgradTransposeMLeadingDim) {
  if (!SupportsCudnnRaggedDotWgrad()) {
    GTEST_SKIP() << "CuDNN ragged dot wgrad requires cuDNN 9.24+.";
  }

  const auto& [data_type, group_type] = GetParam();
  const std::string hlo_with_new_type =
      absl::StrReplaceAll(R"(
    HloModule Test

    ENTRY Test {
      input = TYPE[510,128]{1,0} parameter(0)
      doutput = TYPE[254,128]{1,0} parameter(1)
      group_sizes = GROUP_TYPE[16]{0} constant({8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8})
      ROOT rd = TYPE[16,510,254]{2,1,0} ragged-dot(input, doutput, group_sizes),
             lhs_contracting_dims={1}, rhs_contracting_dims={1}, lhs_ragged_dims={1}
    })",
                          {{"TYPE", data_type}, {"GROUP_TYPE", group_type}});
  std::string optimized_hlo_string = GetOptimizedHlo(hlo_with_new_type);
  EXPECT_THAT(optimized_hlo_string, HasSubstr(kCuDnnFusionKind));

  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(hlo_with_new_type));
  DebugOptions debug_opts = module->config().debug_options();
  debug_opts.set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  module->mutable_config().set_debug_options(debug_opts);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}))
      << optimized_hlo_string;
}

INSTANTIATE_TEST_SUITE_P(AllTypes, RaggedDotFusionRewriterIntegrationTest,
                         ::testing::Combine(::testing::ValuesIn(kbf16f16),
                                            ::testing::ValuesIn(ks32s64)));
}  // namespace
}  // namespace gpu
}  // namespace xla
