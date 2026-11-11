/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/ragged_dot_rewriter.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class RaggedDotRewriterTest : public HloHardwareIndependentTestBase {
 protected:
  se::GpuComputeCapability GetCudaComputeCapability() {
    return se::CudaComputeCapability{se::CudaComputeCapability::kAmpere, 0};
  }

  se::GpuComputeCapability GetRocmComputeCapability() {
    return se::GpuComputeCapability(se::RocmComputeCapability{"gfx942"});
  }

  // Runs the rewriter; if `expected` is present, checks it with FileCheck on
  // the output, otherwise checks that the module has not changed.
  void CheckRaggedDotRewrite(absl::string_view hlo,
                             std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(
        hlo, RaggedDotRewriter(GetCudaComputeCapability()), expected);
  }
};

TEST_F(RaggedDotRewriterTest, DontDoAnythingIfNoRaggedDot) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,4]{1,0} parameter(0)
    p1 = f32[64,4]{1,0} parameter(1)
    ROOT dot = f32[64]{0} dot(p0, p1), lhs_batch_dims={0},
                                       lhs_contracting_dims={1},
                                       rhs_batch_dims={0},
                                       rhs_contracting_dims={1}
  })",
                        std::nullopt);
}

TEST_F(RaggedDotRewriterTest, DontRewriteIfUsingRaggedDotFusion) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RaggedDotRewriter(GetCudaComputeCapability(),
                                            kMinCudnnVersionForRaggedDotFusion)
                              .Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedDotRewriterTest, RewriteIfPreAmpereForRaggedDotFusion) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  // Volta is pre-Ampere, so even with a recent cuDNN the fusion gate must
  // fail and the rewriter must fall back to expansion.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RaggedDotRewriter(
          se::CudaComputeCapability{se::CudaComputeCapability::kVolta, 0},
          kMinCudnnVersionForRaggedDotFusion)
          .Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(RaggedDotRewriterTest, RewriteIfCudnnVersionTooOldForRaggedDotFusion) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RaggedDotRewriter(GetCudaComputeCapability(), se::dnn::VersionInfo{9, 21})
          .Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(RaggedDotRewriterTest, RaggedDotWgradFusionGate) {
  constexpr absl::string_view kWgradModule = R"hlo(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[64,8]{1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[2,9,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={0}, rhs_contracting_dims={0},
                      lhs_ragged_dims={0}
  })hlo";

  struct TestCase {
    se::GpuComputeCapability cc;
    se::dnn::VersionInfo cudnn;
    se::SemanticVersion toolkit;
    bool expected_changed;
  };
  const TestCase cases[] = {
      {se::CudaComputeCapability::Hopper(),
       kMinCudnnVersionForRaggedDotWgradFusion,
       kMinCudaToolkitVersionForRaggedDotWgradFusion, false},
      {se::CudaComputeCapability::Blackwell(),
       se::dnn::VersionInfo{9, 25, 1},
       {13, 3, 0},
       false},
      {se::CudaComputeCapability::Hopper(),
       se::dnn::VersionInfo{9, 25, 1},
       {13, 2, 0},
       true},
      {se::CudaComputeCapability::Hopper(),
       se::dnn::VersionInfo{9, 23, 0},
       {13, 3, 0},
       true},
      {se::CudaComputeCapability::Ampere(),
       se::dnn::VersionInfo{9, 25, 1},
       {13, 3, 0},
       true},
      {se::CudaComputeCapability{se::CudaComputeCapability::kBlackwell_12, 0},
       se::dnn::VersionInfo{9, 25, 1},
       {13, 3, 0},
       true},
  };
  for (const TestCase& tc : cases) {
    EXPECT_EQ(SupportsCudnnRaggedDotWgrad(tc.cc, tc.cudnn, tc.toolkit),
              !tc.expected_changed);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(kWgradModule));
    module->mutable_config()
        .mutable_debug_options()
        .set_xla_gpu_experimental_use_ragged_dot_fusion(true);
    ASSERT_OK_AND_ASSIGN(
        bool changed,
        RaggedDotRewriter(tc.cc, tc.cudnn, tc.toolkit).Run(module.get()));
    EXPECT_EQ(changed, tc.expected_changed);
  }
  EXPECT_FALSE(SupportsCudnnRaggedDotWgrad(
      GetRocmComputeCapability(), se::dnn::VersionInfo{9, 25, 1}, {13, 3, 0}));
}

TEST_F(RaggedDotRewriterTest, DontRewriteIfUsingRaggedDotFusionRocm) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_use_ragged_dot_fusion(true);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RaggedDotRewriter(GetRocmComputeCapability()).Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedDotRewriterTest, DontRewriteIfUsingRaggedDotGroupedGemm) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  auto& debug_option = module->mutable_config().mutable_debug_options();
  debug_option.set_xla_gpu_experimental_use_ragged_dot_grouped_gemm(true);
  debug_option.set_xla_gpu_enable_cublaslt(true);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RaggedDotRewriter(GetRocmComputeCapability()).Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(RaggedDotRewriterTest,
       RewriteIfUsingRaggedDotGroupedGemmAndUnsupportedDatatype) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,9]{1,0} parameter(0)
    p1 = f32[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = f32[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  auto& debug_option = module->mutable_config().mutable_debug_options();
  debug_option.set_xla_gpu_experimental_use_ragged_dot_grouped_gemm(true);
  debug_option.set_xla_gpu_enable_cublaslt(true);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RaggedDotRewriter(GetRocmComputeCapability()).Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(RaggedDotRewriterTest,
       RewriteIfUsingRaggedDotGroupedGemmWithCudaComputeCapability) {
  absl::string_view module_string = R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_string));
  auto& debug_option = module->mutable_config().mutable_debug_options();
  debug_option.set_xla_gpu_experimental_use_ragged_dot_grouped_gemm(true);
  debug_option.set_xla_gpu_enable_cublaslt(true);
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RaggedDotRewriter(GetCudaComputeCapability()).Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(RaggedDotRewriterTest, RaggedNonContracting) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = bf16[64,9]{1,0} parameter(0)
    p1 = bf16[2,9,8]{2,1,0} parameter(1)
    p2 = s64[2]{0} parameter(2)
    ROOT ragged-dot = bf16[64,8]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = bf16[64,8]{1,0}
  // CHECK-SAME: lhs_contracting_dims={0,2}, rhs_contracting_dims={0,1}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedNonContractingWithBatchDimensions) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = bf16[3,64,9]{2,1,0} parameter(0)
    p1 = bf16[3,2,9,8]{3,2,1,0} parameter(1)
    p2 = s64[3,2]{1,0} parameter(2)
    ROOT ragged-dot = bf16[3,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={2},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={1}, rhs_group_dims={1}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = bf16[3,64,8]{2,1,0}
  // CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={1,3},
  // CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={1,2}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedNonContractingWithNonLeadingBatchDim) {
  // The batch dimension is not the leading one, so the group dimension the
  // expansion inserts shifts it.
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[64,2,16]{2,1,0} parameter(0)
    p1 = f32[3,2,16,8]{3,2,1,0} parameter(1)
    p2 = s32[2,3]{1,0} parameter(2)
    ROOT ragged-dot = f32[2,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={2},
                      lhs_batch_dims={1}, rhs_batch_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
  })",
                        R"(
  // CHECK: s32[64,3,2,16]{3,2,1,0} broadcast({{.*}}), dimensions={2,1}
  // CHECK: f32[64,3,2,16]{3,2,1,0} broadcast({{.*}}), dimensions={0,2,3}
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[2,64,8]{2,1,0}
  // CHECK-SAME: lhs_batch_dims={2}, lhs_contracting_dims={1,3},
  // CHECK-SAME: rhs_batch_dims={1}, rhs_contracting_dims={0,2}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedNonContractingWithFlatGroupSizes) {
  // group_sizes is a flat [G] shared by every batch element, rather than
  // carrying one row per batch element.
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[3,64,16]{2,1,0} parameter(0)
    p1 = f32[3,2,16,8]{3,2,1,0} parameter(1)
    p2 = s32[2]{0} parameter(2)
    ROOT ragged-dot = f32[3,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={2},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={1}, rhs_group_dims={1}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[3,64,8]{2,1,0}
  // CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={1,3},
  // CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={1,2}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedNonContractingWithExtraLhsDim) {
  // group_sizes ranges over the lhs non-contracting dimension outer to the
  // ragged one, not over a batch dimension.
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[2,6,4]{2,1,0} parameter(0)
    p1 = f32[3,4,5]{2,1,0} parameter(1)
    p2 = s64[2,3]{1,0} parameter(2)
    ROOT ragged-dot = f32[2,6,5]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={1}, rhs_group_dims={0}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[2,6,5]{2,1,0}
  // CHECK-SAME: lhs_contracting_dims={1,3}, rhs_contracting_dims={0,1}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContractingWithExtraContractingDim) {
  // group_sizes ranges over the contracting dimension outer to the ragged
  // one, which the group dimension the expansion inserts shifts.
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[2,11,3,5]{3,2,1,0} parameter(0)
    p1 = f32[2,7,3,5]{3,2,1,0} parameter(1)
    p2 = s64[2,3,4]{2,1,0} parameter(2)
    ROOT ragged-dot = f32[4,2,11,7]{3,2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2,3}, rhs_contracting_dims={2,3},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={3}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[4,2,11,7]{3,2,1,0}
  // CHECK-SAME: lhs_batch_dims={0,1}, lhs_contracting_dims={3,4},
  // CHECK-SAME: rhs_batch_dims={0,1}, rhs_contracting_dims={3,4}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContracting) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[11,5]{1,0} parameter(0)
    p1 = f32[5,7]{1,0} parameter(1)
    p2 = s32[3]{0} parameter(2)
    ROOT ragged-dot = f32[3,11,7]{2,1,0} ragged-dot(p0, p1, p2),
                                      lhs_contracting_dims={1},
                                      rhs_contracting_dims={0},
                                      lhs_ragged_dims={1}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[3,11,7]{2,1,0}
  // CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2},
  // CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={1}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContractingWithBatchDims) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[2,11,5]{2,1,0} parameter(0)
    p1 = f32[2,5,7]{2,1,0} parameter(1)
    p2 = s32[2,3]{1,0} parameter(2)
    ROOT ragged-dot = f32[3,2,11,7]{3,2,1,0} ragged-dot(p0, p1, p2),
                                      lhs_contracting_dims={2},
                                      rhs_contracting_dims={1},
                                      lhs_batch_dims={0},
                                      rhs_batch_dims={0},
                                      lhs_ragged_dims={2}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[3,2,11,7]{3,2,1,0}
  // CHECK-SAME: lhs_batch_dims={0,1}, lhs_contracting_dims={3},
  // CHECK-SAME: rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContractingWithBatchDimsAtDifferentPos) {
  // The lhs and the rhs hold the batch dimension group_sizes ranges over at
  // different positions.
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[5,2,6]{2,1,0} parameter(0)
    p1 = f32[2,6,7]{2,1,0} parameter(1)
    p2 = s32[2,3]{1,0} parameter(2)
    ROOT ragged-dot = f32[3,2,5,7]{3,2,1,0} ragged-dot(p0, p1, p2),
                                      lhs_contracting_dims={2},
                                      rhs_contracting_dims={1},
                                      lhs_batch_dims={1},
                                      rhs_batch_dims={0},
                                      lhs_ragged_dims={2}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[3,2,5,7]{3,2,1,0}
  // CHECK-SAME: lhs_batch_dims={0,2}, lhs_contracting_dims={3},
  // CHECK-SAME: rhs_batch_dims={0,1}, rhs_contracting_dims={2}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContractingWithMultipleBatchDims) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[2,4,11,5]{3,2,1,0} parameter(0)
    p1 = f32[2,4,5,7]{3,2,1,0} parameter(1)
    p2 = s32[2,4,6]{2,1,0} parameter(2)
    ROOT ragged-dot = f32[6,2,4,11,7]{4,3,2,1,0} ragged-dot(p0, p1, p2),
                                      lhs_contracting_dims={3},
                                      rhs_contracting_dims={2},
                                      lhs_batch_dims={0,1},
                                      rhs_batch_dims={0,1},
                                      lhs_ragged_dims={3}
  })",
                        R"(
  // CHECK: f32[6,2,4,11,5]{4,3,2,1,0} transpose({{.*}}), dimensions={2,0,1,3,4}
  // CHECK: f32[6,2,4,5,7]{4,3,2,1,0} transpose({{.*}}), dimensions={2,0,1,3,4}
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[6,2,4,11,7]{4,3,2,1,0}
  // CHECK-SAME: lhs_batch_dims={0,1,2}, lhs_contracting_dims={4},
  // CHECK-SAME: rhs_batch_dims={0,1,2}, rhs_contracting_dims={3}
  )");
}

TEST_F(RaggedDotRewriterTest, RaggedContractingWithMultipleContractingDims) {
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = f32[2,11,3,5]{3,2,1,0} parameter(0)
    p1 = f32[2,7,3,5]{3,2,1,0} parameter(1)
    p2 = s32[6]{0} parameter(2)
    ROOT ragged-dot = f32[6,2,11,7]{3,2,1,0} ragged-dot(p0, p1, p2),
                                      lhs_contracting_dims={2,3},
                                      rhs_contracting_dims={2,3},
                                      lhs_batch_dims={0},
                                      rhs_batch_dims={0},
                                      lhs_ragged_dims={3}
  })",
                        R"(
  // CHECK: ROOT [[dot:%[^ ]+]] = f32[6,2,11,7]{3,2,1,0}
  // CHECK-SAME: lhs_batch_dims={0,1}, lhs_contracting_dims={3,4},
  // CHECK-SAME: rhs_batch_dims={0,1}, rhs_contracting_dims={3,4}
  )");
}

TEST_F(RaggedDotRewriterTest, BatchContracting) {
  // A ragged batch dimension needs no masking: the ragged-dot becomes a plain
  // dot on the original operands.
  CheckRaggedDotRewrite(R"(
  HloModule module

  ENTRY main {
    p0 = bf16[3,64,9]{2,1,0} parameter(0)
    p1 = bf16[3,9,8]{2,1,0} parameter(1)
    p2 = s64[3]{0} parameter(2)
    ROOT ragged-dot = bf16[3,64,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={0}
  })",
                        R"(
  // CHECK-NOT: select
  // CHECK: ROOT [[dot:%[^ ]+]] = bf16[3,64,8]{2,1,0} dot(
  // CHECK-SAME: p0,
  // CHECK-SAME: p1)
  // CHECK-SAME: lhs_batch_dims={0}, lhs_contracting_dims={2},
  // CHECK-SAME: rhs_batch_dims={0}, rhs_contracting_dims={1}
  )");
}

}  // namespace
}  // namespace xla
