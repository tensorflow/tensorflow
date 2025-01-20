/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/gemm_fusion.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/cublas_padding_requirements.h"
#include "xla/service/gpu/triton_fusion_analysis.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;

namespace m = ::xla::match;

class GemmFusionTest : public HloTestBase {
 public:
  GemmFusionTest()
      : HloTestBase(/*verifier_layout_sensitive=*/true,
                    /*allow_mixed_precision_in_hlo_verifier=*/false) {}

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_gemm_any(false);
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    return debug_options;
  }

  se::GpuComputeCapability gpu_version_{
      se::CudaComputeCapability{se::CudaComputeCapability::AMPERE, 0}};

  void MatchHloModule(HloModule& module, absl::string_view pattern) {
    TF_ASSERT_OK_AND_ASSIGN(bool filecheck_result,
                            RunFileCheck(module.ToString(), pattern));
    EXPECT_TRUE(filecheck_result);
  }
};

TEST_F(GemmFusionTest, TransposeSubdimensionGroup) {
  // This HLO is artificial because unnecessary reshapes get optimized
  // out during compilation. It tests the ability of GemmFusion
  // to handle transposes of groups of subdimensions.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f32[32,3] parameter(0)
  t1 = f32[3,32] transpose(p0), dimensions={1,0}
  r1 = f32[3,8,4] reshape(t1)
  r0 = f32[3,32] reshape(r1)
  p1 = f16[32,7] parameter(1)
  c1 = f32[32,7] convert(p1)
  ROOT d = f32[3,7] dot(r0, c1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionTest, UnsupportedTransposeIsNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[1,512,8,1024]{3,1,0,2} parameter(0)
  c = f16[1,512,8,1024]{3,2,1,0} copy(p0)
  b = f16[4096,1024]{1,0} bitcast(c)
  p1 = f16[128,1024]{1,0} parameter(1)
  ROOT d = f16[4096,128]{1,0} dot(b, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})")
                    .value();
  EXPECT_FALSE(GemmFusion(gpu_version_).Run(module.get()).value());
}

TEST_F(GemmFusionTest, BitcastChain) {
  // This HLO is artificial because unnecessary reshapes get optimized
  // out during compilation. It tests the ability of GemmFusion
  // to handle various kinds of bitcasts.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = s8[60,5] parameter(0)
  r0 = s8[3,20,5] reshape(p0)
  c0 = f16[3,20,5] convert(r0)
  p1 = f16[3,200] parameter(1)
  r12 = f16[600] reshape(p1)
  r11 = f16[30,20] reshape(r12)
  r1 = f16[3,10,20] reshape(r11)
  ROOT d = f16[3,5,10] dot(c0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionTest, SplitDimensionTwice) {
  auto module = ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[4,2,32,4,2] parameter(0)
  r1 = s8[8,32,8] reshape(p0)
  t1 = s8[32,8,8] transpose(r1), dimensions={1,0,2}
  r0 = s8[32,64] reshape(t1)
  p1 = s8[32,32] parameter(1)
  c0 = f16[32,32] convert(p1)
  ROOT d = f16[64,32] dot(r0, c0),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionTest, DoNotTriggerOnUnsupportedOutputConversions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f16[128,256] parameter(0)
  p1 = f16[256,512] parameter(1)
  r = f16[128,512] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT c = u8[128,512] convert(r)
})"));
  EXPECT_FALSE(GemmFusion(gpu_version_).Run(module.get()).value());
}

TEST_F(GemmFusionTest, FuseDotWithTrivialNoncontractingDim) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = s8[60,5] parameter(0)
  r0 = s8[3,20,5] reshape(p0)
  c0 = f16[3,20,5] convert(r0)
  p1 = f16[3,1,20] parameter(1)
  ROOT d = f16[3,5,1] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={2},
    lhs_batch_dims={0}, rhs_batch_dims={0}
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionTest, HandleDotIfCublasRequiresPadding) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f16[5,3] parameter(0)
  p1 = f16[5,7] parameter(1)
  ROOT d = f16[3,7] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(CublasRequiresPadding(
      *xla::Cast<HloDotInstruction>(
          module->entry_computation()->root_instruction()),
      cc));
  EXPECT_TRUE(GemmFusion(cc).Run(module.get()).value());
}

TEST_F(GemmFusionTest, FuseSliceOfParameterWithOtherUsers) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[97,121] parameter(0)
  s0 = f32[7,101] slice(p0), slice={[3:10], [10:111]}
  p1 = f32[101,16] parameter(1)
  d = f32[16,7] dot(p1, s0),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
  sout1 = f32[3,33] slice(p0), slice={[10:13], [20:53]}
  ROOT t = tuple(d, sout1)
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(GemmFusion(cc).Run(module.get()).value());
}

TEST_F(GemmFusionTest, DoNotFuseSliceOfMixedDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = bf16[768,64] parameter(0)
  s0 = bf16[768,32] slice(p0), slice={[0:768], [0:32]}
  b0 = bf16[256,3,32] reshape(s0)
  b1 = bf16[256,96] reshape(b0)
  p1 = bf16[256,96] parameter(1)
  ROOT d = bf16[96,96] dot(b1, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_FALSE(GemmFusion(cc).Run(module.get()).value());
}

TEST_F(GemmFusionTest, DoNotFuseSlicesOfNonMajorFragments) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[2,2,256,256] parameter(0)
  s0 = f32[1,1,256,256] slice(p0),
    slice={[0:1], [0:1], [0:256], [0:256]}
  r0 = f32[256,256] reshape(s0)
  p1 = f16[2,2,256,256] parameter(1)
  sout1 = f16[1,1,256,256] slice(p1),
    slice={[0:1], [0:1], [0:256], [0:256]}
  r1 = f16[256,256] reshape(sout1)
  ROOT d = f32[256,256] dot(r0, r1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));

  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_FALSE(GemmFusion(cc).Run(module.get()).value());
}

TEST_F(GemmFusionTest, DynamicSliceIsFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  dot_lhs = f32[2,18] parameter(0)
  dynamic_slice_input = f32[2,64,2] parameter(1)
  start_index0 = s32[] parameter(2)
  start_index1_2 = s32[] constant(0)
  dynamic_slice = f32[1,64,2] dynamic-slice(dynamic_slice_input, start_index0, start_index1_2, start_index1_2),
                  dynamic_slice_sizes={1,64,2}
  reshape = f32[64,2] reshape(dynamic_slice)
  ROOT dot = f16[18,64] dot(dot_lhs, reshape),
             lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Parameter(), m::Constant()))));
}

TEST_F(GemmFusionTest, DynamicSlicesAreFusedEvenIfTheyShareIndices) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[2,64,2] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  p3 = s32[] parameter(3)
  ds0 = f32[1,64,2] dynamic-slice(p0, p1, p2, p3), dynamic_slice_sizes={1,64,2}
  a = f32[64,2] reshape(ds0)
  ds1 = f32[1,64,2] dynamic-slice(p0, p3, p2, p1), dynamic_slice_sizes={1,64,2}
  b = f32[64,2] reshape(ds1)
  ROOT d = f16[64,64] dot(a, b),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  // TODO(b/339810582): Don't duplicate scalar parameters to dot fusions,
  // because they are never tiled differently.
  // TODO(b/339814210): Don't count scalar parameters towards dot fusion
  // parameter limit.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch((m::Fusion(m::Parameter(), m::Parameter(), m::Parameter(),
                            m::Parameter(), m::Parameter(), m::Parameter(),
                            m::Parameter(), m::Parameter()))));
}

TEST_F(GemmFusionTest, DoNotFuseDynamicSliceOfNonMajorFragments) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  dot_lhs = f32[2,4]{1,0} parameter(0)
  dynamic_slice_input = f32[4,5,2]{2,1,0} parameter(1)
  c0 = s32[] constant(0)
  c2 = s32[] constant(2)
  dynamic_slice = f32[4,1,2]{2,1,0} dynamic-slice(dynamic_slice_input, c0, c2, c0),
                  dynamic_slice_sizes={4,1,2}
  reshape = f32[4,2]{1,0} reshape(dynamic_slice)
  ROOT dot = f32[4,4]{1,0} dot(dot_lhs, reshape),
             lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  // FusionDecision "Unsupported dynamic slice on non-major-most dimension."
  EXPECT_FALSE(GemmFusion(cc).Run(module.get()).value());
}

TEST_F(GemmFusionTest, CanFuseDynamicSliceOfContractingDimIfItIsMajor) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  dot_lhs = f32[2,4]{1,0} parameter(0)
  dynamic_slice_input = f32[5,5]{1,0} parameter(1)
  start_index0 = s32[] constant(2)
  start_index1 = s32[] constant(0)
  dynamic_slice = f32[2,5]{1,0} dynamic-slice(dynamic_slice_input, start_index0, start_index1),
                  dynamic_slice_sizes={2,5}
  ROOT d = f32[4,5]{1,0} dot(dot_lhs, dynamic_slice),
           lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Constant(), m::Constant()))));
}

TEST_F(GemmFusionTest, SliceToDegenerateIsSkipped) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p = f32[3] parameter(0)
  s = f32[1] slice(p), slice={[2:3]}
  r = f32[] reshape(s)
  b = f32[3,3] broadcast(r), dimensions={}
  ROOT d = f32[3,3] dot(b, b),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};

  ASSERT_TRUE(GemmFusion(cc).Run(module.get()).value());

  // Slice is not fused.
  MatchHloModule(*module, R"(
; CHECK-NOT: slice
; CHECK: ENTRY
; CHECK: slice
)");
}

TEST_F(GemmFusionTest, MultipleUsesAreHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  c = f32[] constant(1)
  b = f32[6,8] broadcast(c), dimensions={}
  p0 = f32[6,8] parameter(0)
  a1 = f32[6,8] add(p0, b)
  e = f32[6,8] exponential(a1)
  a2 = f32[6,8] add(e, b)
  d = f32[6,8] divide(b, a2)
  p2 = f16[8,6] parameter(1)
  cv = f32[8,6] convert(p2)
  ROOT r = f32[6,6] dot(d, cv),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(GemmFusion(cc).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionTest, BinaryElementwiseOfBroadcastIsFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p2 = f32[3072] parameter(2)
  b = f32[8192,3072] broadcast(p2), dimensions={1}
  p0 = f16[8192,3072] parameter(0)
  p0c = f32[8192,3072] convert(p0)
  a = f32[8192,3072] add(p0c, b)
  p1 = f32[3072,768] parameter(1)
  ROOT r = f32[8192,768] dot(a, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(GemmFusion(cc).Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionTest, BinaryElementwiseOfUnsupportedBroadcastIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p2 = f32[768] parameter(2)
  b = f32[8192,768,4] broadcast(p2), dimensions={1}
  s = f32[8192,3072] bitcast(b)
  p0 = f16[8192,3072] parameter(0)
  p0c = f32[8192,3072] convert(p0)
  a = f32[8192,3072] add(p0c, s)
  p1 = f32[3072,768] parameter(1)
  ROOT r = f32[8192,768] dot(a, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_FALSE(GemmFusion(cc).Run(module.get()).value());
}

class GemmFusionLevel2Test : public GemmFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmFusionTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_triton_fusion_level(2);
    return debug_options;
  }
};

TEST_F(GemmFusionTest, ConcatenationDivisibleBy64IsFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = bf16[8192,1]{1,0} parameter(0)
  p1 = bf16[2752,8192]{1,0} parameter(1)
  p2 = bf16[2752,8192]{1,0} parameter(2)
  concat = bf16[5504,8192]{1,0} concatenate(p1, p2), dimensions={0}
  bitcast = bf16[8192,5504]{0,1} bitcast(concat)
  ROOT r = f32[1,5504]{1,0} dot(p0, bitcast),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));
  const se::CudaComputeCapability cc{se::CudaComputeCapability::AMPERE, 0};
  EXPECT_TRUE(GemmFusion(cc).Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Parameter(), m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionLevel2Test, ReshapeToScalarIsHandled) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[5,3] parameter(0)
  c = f16[5,3] convert(p0)
  p1 = f16[1] parameter(1)
  r = f16[] reshape(p1)
  b = f16[5,7] broadcast(r)
  ROOT d = f16[3,7] dot(c, b),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionLevel2Test, DoNotFuseIncompatibleDimensionSplits) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p1 = s8[5,7,2,3]{3,2,1,0} parameter(1)
  t1 = s8[7,5,2,3]{3,2,1,0} transpose(p1), dimensions={1,0,2,3}
  r1 = s8[7,30]{1,0} reshape(t1)
  cvt = f16[7,30]{1,0} convert(r1)
  p2 = f16[2,7,5,3]{3,2,1,0} parameter(2)
  t2 = f16[7,2,5,3]{3,2,1,0} transpose(p2), dimensions={1,0,2,3}
  r2 = f16[7,30]{1,0} reshape(t2)
  a = f16[7,30]{1,0} add(cvt, r2)
  p0 = f16[7,79]{1,0} parameter(0)
  ROOT dot = f16[30,79]{1,0} dot(a, p0),
    lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Fusion(m::Transpose(), m::Parameter(), m::Parameter())));
}

TEST_F(GemmFusionLevel2Test, DoNotFuseTooManyParameters) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  tmp_0 = f32[] constant(1)
  tmp_1 = f32[3,49]{1,0} broadcast(tmp_0), dimensions={}
  tmp_2 = f32[3,49]{1,0} parameter(6)
  tmp_3 = f32[] constant(0)
  tmp_4 = f32[3,49]{1,0} broadcast(tmp_3), dimensions={}
  tmp_5 = pred[3,49]{1,0} compare(tmp_2, tmp_4), direction=GT
  tmp_6 = f32[3,49]{1,0} convert(tmp_5)
  tmp_7 = f32[3,49]{1,0} subtract(tmp_1, tmp_6)
  tmp_8 = s32[] parameter(13)
  tmp_9 = f32[] convert(tmp_8)
  tmp_10 = f32[] maximum(tmp_9, tmp_0)
  tmp_11 = f32[] divide(tmp_3, tmp_10)
  tmp_12 = f32[3,49]{1,0} broadcast(tmp_11), dimensions={}
  tmp_13 = pred[3,49]{1,0} parameter(7)
  tmp_14 = pred[3,49]{1,0} parameter(10)
  tmp_15 = pred[3,49]{1,0} and(tmp_13, tmp_14)
  tmp_16 = f32[3,49]{1,0} convert(tmp_15)
  tmp_17 = f32[3,49]{1,0} multiply(tmp_12, tmp_16)
  tmp_18 = f32[3,49]{1,0} negate(tmp_17)
  tmp_19 = f32[3,49]{1,0} multiply(tmp_7, tmp_18)
  tmp_20 = f32[3,49]{1,0} parameter(19)
  tmp_21 = f32[3,49]{1,0} subtract(tmp_1, tmp_20)
  tmp_22 = f32[3,49]{1,0} divide(tmp_19, tmp_21)
  tmp_23 = f32[3,49]{1,0} negate(tmp_22)
  tmp_24 = f32[3,49]{1,0} negate(tmp_6)
  tmp_25 = f32[3,49]{1,0} multiply(tmp_24, tmp_17)
  tmp_26 = f32[3,49]{1,0} divide(tmp_25, tmp_20)
  tmp_27 = f32[3,49]{1,0} add(tmp_23, tmp_26)
  tmp_28 = f32[3,49]{1,0} parameter(18)
  tmp_29 = f32[3,49]{1,0} multiply(tmp_27, tmp_28)
  tmp_30 = f32[3,49]{1,0} parameter(17)
  tmp_31 = f32[3,49]{1,0} multiply(tmp_29, tmp_30)
  tmp_32 = f32[3,49]{1,0} parameter(16)
  tmp_33 = f32[3,49]{1,0} multiply(tmp_31, tmp_32)
  tmp_34 = f32[3,49]{1,0} parameter(15)
  tmp_35 = f32[3,49]{1,0} add(tmp_33, tmp_34)
  tmp_36 = f32[3,49]{1,0} parameter(14)
  tmp_37 = f32[3,49]{1,0} add(tmp_35, tmp_36)
  tmp_38 = f32[1,1]{1,0} constant({ {0} })
  tmp_39 = f32[1,1]{1,0} broadcast(tmp_38), dimensions={0,1}
  tmp_40 = f32[] reshape(tmp_39)
  tmp_41 = f32[3,32]{1,0} broadcast(tmp_40), dimensions={}
  tmp_42 = u32[48]{0} parameter(11)
  tmp_43 = u32[48]{0} parameter(5)
  tmp_44 = u32[96]{0} concatenate(tmp_42, tmp_43), dimensions={0}
  tmp_45 = u32[3,32]{1,0} reshape(tmp_44)
  tmp_46 = u32[96]{0} reshape(tmp_45)
  tmp_47 = u32[] constant(1)
  tmp_48 = u32[3,32]{1,0} broadcast(tmp_47), dimensions={}
  tmp_49 = u32[96]{0} reshape(tmp_48)
  tmp_50 = u32[96]{0} shift-right-logical(tmp_46, tmp_49)
  tmp_51 = u32[3,32]{1,0} reshape(tmp_50)
  tmp_52 = u32[3,32]{1,0} or(tmp_51, tmp_48)
  tmp_53 = f32[3,32]{1,0} bitcast-convert(tmp_52)
  tmp_54 = f32[3,32]{1,0} broadcast(tmp_0), dimensions={}
  tmp_55 = f32[3,32]{1,0} subtract(tmp_53, tmp_54)
  tmp_56 = f32[1,1]{1,0} constant({ {1} })
  tmp_57 = f32[1,1]{1,0} broadcast(tmp_56), dimensions={0,1}
  tmp_58 = f32[] reshape(tmp_57)
  tmp_59 = f32[3,32]{1,0} broadcast(tmp_58), dimensions={}
  tmp_60 = f32[3,32]{1,0} multiply(tmp_55, tmp_59)
  tmp_61 = f32[3,32]{1,0} add(tmp_60, tmp_41)
  tmp_62 = f32[3,32]{1,0} maximum(tmp_41, tmp_61)
  tmp_63 = f32[3,32]{1,0} broadcast(tmp_3), dimensions={}
  tmp_64 = pred[3,32]{1,0} compare(tmp_62, tmp_63), direction=LT
  tmp_65 = f32[3,32]{1,0} convert(tmp_64)
  tmp_66 = f32[3,49]{1,0} parameter(9)
  tmp_67 = f32[49]{0} parameter(4)
  tmp_68 = f32[3,49]{1,0} broadcast(tmp_67), dimensions={1}
  tmp_69 = f32[3,49]{1,0} add(tmp_66, tmp_68)
  tmp_70 = f32[1,49]{1,0} parameter(12)
  tmp_71 = f32[1,49]{1,0} broadcast(tmp_0), dimensions={}
  tmp_72 = f32[1,49]{1,0} divide(tmp_70, tmp_71)
  tmp_73 = f32[1,49]{1,0} broadcast(tmp_72), dimensions={0,1}
  tmp_74 = f32[49]{0} reshape(tmp_73)
  tmp_75 = f32[3,49]{1,0} broadcast(tmp_74), dimensions={1}
  tmp_76 = f32[3,49]{1,0} subtract(tmp_69, tmp_75)
  tmp_77 = f32[1,49]{1,0} parameter(3)
  tmp_78 = f32[1,49]{1,0} parameter(8)
  tmp_79 = f32[1,49]{1,0} divide(tmp_78, tmp_71)
  tmp_80 = f32[1,49]{1,0} multiply(tmp_72, tmp_72)
  tmp_81 = f32[1,49]{1,0} subtract(tmp_79, tmp_80)
  tmp_82 = f32[1,49]{1,0} add(tmp_81, tmp_71)
  tmp_83 = f32[1,49]{1,0} rsqrt(tmp_82)
  tmp_84 = f32[1,49]{1,0} multiply(tmp_77, tmp_83)
  tmp_85 = f32[1,49]{1,0} broadcast(tmp_84), dimensions={0,1}
  tmp_86 = f32[49]{0} reshape(tmp_85)
  tmp_87 = f32[3,49]{1,0} broadcast(tmp_86), dimensions={1}
  tmp_88 = f32[3,49]{1,0} multiply(tmp_76, tmp_87)
  tmp_89 = f32[1,49]{1,0} parameter(2)
  tmp_90 = f32[1,49]{1,0} broadcast(tmp_89), dimensions={0,1}
  tmp_91 = f32[49]{0} reshape(tmp_90)
  tmp_92 = f32[3,49]{1,0} broadcast(tmp_91), dimensions={1}
  tmp_93 = f32[3,49]{1,0} add(tmp_88, tmp_92)
  tmp_94 = f32[49,32]{1,0} parameter(1)
  tmp_95 = f32[3,32]{1,0} dot(tmp_93, tmp_94), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  tmp_96 = f32[32]{0} parameter(0)
  tmp_97 = f32[3,32]{1,0} broadcast(tmp_96), dimensions={1}
  tmp_98 = f32[3,32]{1,0} add(tmp_95, tmp_97)
  tmp_99 = f32[3,32]{1,0} multiply(tmp_65, tmp_98)
  tmp_100 = f32[3,32]{1,0} divide(tmp_99, tmp_63)
  tmp_101 = f32[3,32]{1,0} maximum(tmp_100, tmp_63)
  ROOT tmp_102 = f32[49,32]{1,0} dot(tmp_37, tmp_101), lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
  EXPECT_EQ(module->entry_computation()->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kCustom);
  EXPECT_LE(module->entry_computation()->root_instruction()->operand_count(),
            TritonFusionAnalysis::kMaxParameterPerDotOperand * 2);
}

TEST_F(GemmFusionLevel2Test,
       DoNotFuseTooManyParametersWhenAnInstructionWouldAddMultipleParameters) {
  static_assert(TritonFusionAnalysis::kMaxParameterPerDotOperand == 4,
                "We have to update this test.");
  // If we fuse the select, it adds 2 additional parameters at once (not 3,
  // because the select instruction itself is removed from the parameters).
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[3,49]{1,0} parameter(0)
  b = f32[3,49]{1,0} parameter(1)
  c = pred[3,49]{1,0} parameter(2)
  d = f32[3,49]{1,0} parameter(3)
  e = f32[3,49]{1,0} parameter(4)
  add0 = f32[3,49]{1,0} add(a, b)
  select = f32[3,49]{1,0} select(c, d, e)
  add1 = f32[3,49]{1,0} add(add0, select)
  f = f32[3,32]{1,0} parameter(5)
  ROOT tmp_102 = f32[49,32]{1,0} dot(add1, f), lhs_contracting_dims={0}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
  EXPECT_EQ(module->entry_computation()->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kCustom);
  EXPECT_LE(module->entry_computation()->root_instruction()->operand_count(),
            TritonFusionAnalysis::kMaxParameterPerDotOperand + 1);
}

TEST_F(GemmFusionLevel2Test, DoNotFuseTooManyParametersForConcat) {
  static_assert(TritonFusionAnalysis::kMaxParameterPerDotOperand == 4,
                "We have to update this test.");
  // The concat shouldn't overgo the allowed parameter limit.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[3,3]{1,0} parameter(0)
  b = f32[3,3]{1,0} parameter(1)
  c = f32[3,3]{1,0} parameter(2)
  d = f32[3,3]{1,0} parameter(3)
  e = f32[3,3]{1,0} parameter(4)
  f = f16[3,3]{1,0} parameter(5)
  concat = f32[15,3]{1,0} concatenate(a, b, c, d, e), dimensions={0}
  convert = f32[3,3]{1,0} convert(f)
  ROOT dot = f32[15,3]{1,0} dot(concat, convert), lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kFusion);
  EXPECT_EQ(module->entry_computation()->root_instruction()->fusion_kind(),
            HloInstruction::FusionKind::kCustom);
  EXPECT_LE(module->entry_computation()->root_instruction()->operand_count(),
            TritonFusionAnalysis::kMaxParameterPerDotOperand + 1);
}

TEST_F(GemmFusionLevel2Test,
       InstructionsReachableFromMultipleOperandsAreHandledCorrectly) {
  static_assert(TritonFusionAnalysis::kMaxParameterPerDotOperand == 4,
                "We have to update this test.");
  // There was a bug that some dead code was generated into some fusions in a
  // specific edge case. When some instructions were reachable both through the
  // LHS and the RHS operands, the BFS (Breadth-first search) through the LHS1
  // operand "marked" one operation as non-fusible because it would exceed the
  // limit on fusion parameters per operand. But the BFS through the RHS operand
  // went through that node and fused some more operands. So the resulting
  // fusion was not connected and caused errors. This test case checks that such
  // configurations generate a correct HLO now.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[2,4]{1,0} parameter(0)
  b = f32[2,4]{1,0} parameter(1)
  c = f32[2,4]{1,0} parameter(2)
  d = f32[2,4]{1,0} parameter(3)
  e = f32[2,4]{1,0} parameter(4)
  add0 = f32[2,4]{1,0} add(a, b)
  add1 = f32[2,4]{1,0} add(add0, c)
  add2 = f32[2,4]{1,0} add(add1, d)
  add3 = f32[2,4]{1,0} add(add2, e)
  ROOT r = f32[2,2]{1,0} dot(add3, add0),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  // ~VerifiedHloModule() will verify the module.
}

TEST_F(GemmFusionLevel2Test, EachScopeIsFusedToASeparateSubgraph) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[2,4]{1,0} parameter(0)
  b = f32[2,4]{1,0} parameter(1)
  add = f32[2,4]{1,0} add(a, b)
  ROOT r = f32[2,2]{1,0} dot(add, add),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
CHECK-DAG: %[[P0:.*]] = f32[2,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[2,4]{1,0} parameter(1)
CHECK-DAG: %[[ADD0:.*]] = f32[2,4]{1,0} add(f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P1]])
CHECK-DAG: %[[P2:.*]] = f32[2,4]{1,0} parameter(2)
CHECK-DAG: %[[P3:.*]] = f32[2,4]{1,0} parameter(3)
CHECK-DAG: %[[ADD1:.*]] = f32[2,4]{1,0} add(f32[2,4]{1,0} %[[P2]], f32[2,4]{1,0} %[[P3]])
CHECK-DAG: ROOT {{.*}} = f32[2,2]{1,0} dot(f32[2,4]{1,0} %[[ADD0]], f32[2,4]{1,0} %[[ADD1]])
CHECK: ENTRY
CHECK-DAG: %[[P0:.*]] = f32[2,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[2,4]{1,0} parameter(1)
CHECK-DAG: ROOT {{.*}} = f32[2,2]{1,0}
CHECK-SAME: fusion(f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P1]], f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P1]]),
CHECK-SAME: kind=kCustom
CHECK-SAME: __triton_gemm
})");
}

// The 2 inputs of the add operation are the same and they are iterated the same
// way, so the same parameter node is reused for them.
// The reuse happens per "operand fusion", so the add of the LHS and RHS still
// use different nodes.
TEST_F(GemmFusionLevel2Test, ParamNodesAreReusedIfTheyHaveTheSameIterSpec) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[2,4]{1,0} parameter(0)
  add = f32[2,4]{1,0} add(a, a)
  ROOT r = f32[2,2]{1,0} dot(add, add),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
CHECK-DAG: %[[P0:.*]] = f32[2,4]{1,0} parameter(0)
CHECK-DAG: %[[ADD0:.*]] = f32[2,4]{1,0} add(f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P0]])
CHECK-DAG: %[[P1:.*]] = f32[2,4]{1,0} parameter(1)
CHECK-DAG: %[[ADD1:.*]] = f32[2,4]{1,0} add(f32[2,4]{1,0} %[[P1]], f32[2,4]{1,0} %[[P1]])
CHECK-DAG: ROOT {{.*}} = f32[2,2]{1,0} dot(f32[2,4]{1,0} %[[ADD0]], f32[2,4]{1,0} %[[ADD1]])
CHECK: ENTRY
CHECK-DAG: %[[P0:.*]] = f32[2,4]{1,0} parameter(0)
CHECK-DAG: ROOT {{.*}} = f32[2,2]{1,0}
CHECK-SAME: fusion(f32[2,4]{1,0} %[[P0]], f32[2,4]{1,0} %[[P0]])
CHECK-SAME: kind=kCustom
CHECK-SAME: __triton_gemm
})");
}

// NEGATE has the same iteration spec at both usages, so the node is reused
// (implying that P0 is also reused).
TEST_F(GemmFusionLevel2Test, NonParamNodesAreReusedIfTheyHaveTheSameIterSpec) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[4,4]{1,0} parameter(0)
  b = f32[4,4]{1,0} parameter(1)
  negate = f32[4,4]{1,0} negate(a)
  sine = f32[4,4]{1,0} sine(negate)
  add = f32[4,4]{1,0} add(negate, sine)
  ROOT r = f32[4,4]{1,0} dot(add, b),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
CHECK-DAG: %[[P0:.*]] = f32[4,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[4,4]{1,0} parameter(1)
CHECK-DAG: %[[NEGATE:.*]] = f32[4,4]{1,0} negate(f32[4,4]{1,0} %[[P0]])
CHECK-DAG: %[[SINE:.*]] = f32[4,4]{1,0} sine(f32[4,4]{1,0} %[[NEGATE]])
CHECK-DAG: %[[ADD:.*]] = f32[4,4]{1,0} add(f32[4,4]{1,0} %[[NEGATE]], f32[4,4]{1,0} %[[SINE]])
CHECK-DAG: ROOT {{.*}} = f32[4,4]{1,0} dot(f32[4,4]{1,0} %[[ADD]], f32[4,4]{1,0} %[[P1]])
CHECK: ENTRY
CHECK-DAG: %[[P0:.*]] = f32[4,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[4,4]{1,0} parameter(1)
CHECK-DAG: ROOT {{.*}} = f32[4,4]{1,0}
CHECK-SAME: fusion(f32[4,4]{1,0} %[[P0]], f32[4,4]{1,0} %[[P1]])
CHECK-SAME: kind=kCustom
CHECK-SAME: __triton_gemm
})");
}

// The direct read of the input and the transposed read of the input have
// different iteration specs, so we don't reuse the node.
TEST_F(GemmFusionLevel2Test, NodesAreNotReusedIfTheyHaveDifferentIterSpecs) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  a = f32[4,4]{1,0} parameter(0)
  b = f32[4,4]{1,0} parameter(1)
  tr_a = f32[4,4]{1,0} transpose(a), dimensions={1,0}
  add = f32[4,4]{1,0} add(a, tr_a)
  ROOT r = f32[4,4]{1,0} dot(add, b),
           lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
CHECK-DAG: %[[P0:.*]] = f32[4,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[4,4]{1,0} parameter(1)
CHECK-DAG: %[[P2:.*]] = f32[4,4]{1,0} parameter(2)
CHECK-DAG: %[[TRANSPOSE:.*]] = f32[4,4]{1,0} transpose(f32[4,4]{1,0} %[[P1]])
CHECK-DAG: %[[ADD:.*]] = f32[4,4]{1,0} add(f32[4,4]{1,0} %[[P0]], f32[4,4]{1,0} %[[TRANSPOSE]])
CHECK-DAG: ROOT {{.*}} = f32[4,4]{1,0} dot(f32[4,4]{1,0} %[[ADD]], f32[4,4]{1,0} %[[P2]])
CHECK: ENTRY
CHECK-DAG: %[[P0:.*]] = f32[4,4]{1,0} parameter(0)
CHECK-DAG: %[[P1:.*]] = f32[4,4]{1,0} parameter(1)
CHECK-DAG: ROOT {{.*}} = f32[4,4]{1,0}
CHECK-SAME: fusion(f32[4,4]{1,0} %[[P0]], f32[4,4]{1,0} %[[P0]], f32[4,4]{1,0} %[[P1]])
CHECK-SAME: kind=kCustom
CHECK-SAME: __triton_gemm
})");
}

TEST_F(GemmFusionLevel2Test, OperationsAddingMoreParametersGetMultipleTries) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = f32[2,2] parameter(0)
  c0 = f32[] constant(12345)
  b0 = f32[2,2] broadcast(c0), dimensions={}
  m0 = f32[2,2] multiply(p0, b0)
  c1 = f32[] constant(34567)
  b1 = f32[2,2] broadcast(c1), dimensions={}
  a0 = f32[2,2] add(m0, b1)
  b3 = f32[2,2,2] broadcast(a0), dimensions={0,1}
  p2 = f32[2,2,2] parameter(2)
  m2 = f32[2,2,2] multiply(p2, b3)
  p1 = f32[2]{0} parameter(1)
  c2 = f32[] constant(5678)
  b2 = f32[2] broadcast(c2), dimensions={}
  a1 = f32[2]{0} add(p1, b2)
  b4 = f32[2,2,2] broadcast(a1), dimensions={2}
  m1 = f32[2,2,2] multiply(m2, b4)
  b = f32[4,2] bitcast(m1)
  p3 = f16[2,2] parameter(3)
  p3c = f32[2,2] convert(p3)
  ROOT r = f32[4,2] dot(b, p3c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Parameter(), m::Parameter()))));
}

TEST_F(GemmFusionLevel2Test, GemmFusionBailsOutPreAmpere) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[2,53] parameter(0)
  p0e = f32[2,53] exponential(p0)
  p1 = s16[53,2] parameter(1)
  p1c = f32[53,2] convert(p1)
  ROOT dot = f32[2,2] dot(p0e, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_THAT(
      GemmFusion(se::CudaComputeCapability{se::CudaComputeCapability::VOLTA, 0})
          .Run(module.get()),
      tsl::testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

TEST_F(GemmFusionLevel2Test, GemmFusionSucceedsOnNonCudaGpu) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[2,53] parameter(0)
  p0e = f32[2,53] exponential(p0)
  p1 = s16[53,2] parameter(1)
  p1c = f32[53,2] convert(p1)
  ROOT dot = f32[2,2] dot(p0e, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmFusion(se::RocmComputeCapability{}).Run(module.get()).ok());
}

TEST_F(GemmFusionLevel2Test, ParameterUsedElementwiseTwiceIsFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

ENTRY e {
  p0 = f32[2,35] parameter(0)
  p0n = f32[2,35] negate(p0)
  p0e = f32[2,35] exponential(p0)
  a = f32[2,35] add(p0e, p0n)
  p1 = f16[35,2] parameter(1)
  p1c = f32[35,2] convert(p1)
  ROOT dot = f32[2,2] dot(a, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter()))));
  TF_ASSERT_OK_AND_ASSIGN(
      const auto analysis,
      TritonFusionAnalysis::Execute(*module->entry_computation()
                                         ->root_instruction()
                                         ->called_computations()[0]));
  EXPECT_EQ(analysis.ScopeParameters(TritonFusionAnalysis::Scope::LHS).size(),
            1);
  EXPECT_EQ(analysis.ScopeParameters(TritonFusionAnalysis::Scope::RHS).size(),
            1);
}

TEST_F(GemmFusionLevel2Test,
       ParameterUsedNonElementwiseTwiceIsFusedOnBothPaths) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule t

ENTRY e {
  p0 = f32[4,4] parameter(0)
  p0t = f32[4,4] transpose(p0), dimensions={1,0}
  a = f32[4,4] add(p0, p0t)
  p1 = f16[4,5] parameter(1)
  p1c = f32[4,5] convert(p1)
  ROOT dot = f32[4,5] dot(a, p1c),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch((m::Fusion(m::Parameter(), m::Parameter(), m::Parameter()))));
}

TEST_F(GemmFusionLevel2Test,
       ComputationParameterWithMultipleUsersIsNotTrivialToFuse) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = f32[400,400] parameter(0)

  c0 = f16[400,400] convert(p0)
  p1 = f16[400,400] parameter(1)
  dot0 = f16[400,400] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}

  c1 = f16[400,400] convert(p0)
  p2 = f16[400,400] parameter(2)
  dot1 = f16[400,400] dot(c1, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}

  ROOT a = f16[400,400] add(dot0, dot1)
})"));
  EXPECT_FALSE(GemmFusion(se::CudaComputeCapability{
                              se::CudaComputeCapability::AMPERE, 0})
                   .Run(module.get())
                   .value());
}

TEST_F(GemmFusionLevel2Test, NarrowingConversionIsAlwaysBetterToFuse) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
ENTRY e {
  p0 = s8[512,512] parameter(0)
  c0 = f16[512,512] convert(p0)
  p1 = f16[512,512] parameter(1)
  dot0 = f16[512,512] dot(c0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}

  n = f16[512,512] negate(c0)
  ROOT a = f16[512,512] add(dot0, n)
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Add(m::Fusion(m::Parameter(), m::Parameter()),
                                 m::Negate()))));
}

TEST_F(GemmFusionLevel2Test, NestedSlicingIsAnalyzedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
triton_gemm_d_computation {
  p0 = f32[6,24]{1,0} parameter(0)
  slice1 = f32[5,20]{1,0} slice(p0), slice={[1:6], [3:23]}
  n1 = f32[5,20]{1,0} negate(slice1)
  slice2 = f32[3,7]{1,0} slice(n1), slice={[1:4], [13:20]}
  p1 = f32[7,37]{1,0} parameter(1)
  ROOT d = f32[3,37]{1,0} dot(slice2, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[7,37]{1,0} parameter(0)
  p1 = f32[6,24]{1,0} parameter(1)
  ROOT triton_gemm_d = f32[3,37]{1,0} fusion(p1, p0), kind=kCustom,
    calls=triton_gemm_d_computation
})"));
  const HloComputation* computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*computation));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS,
                                 computation->parameter_instruction(0), 0),
              ElementsAre(FieldsAre(/*stride=*/24, /*count=*/6,
                                    /*slice_start=*/2, /*sliced_count=*/3,
                                    /*subfragments=*/ElementsAre(3))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::LHS,
                                 computation->parameter_instruction(0), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/24,
                                    /*slice_start=*/16, /*sliced_count=*/7,
                                    /*subfragments=*/ElementsAre(7))));
}

TEST_F(GemmFusionLevel2Test, FusedConcatenationIsAnalyzedCorrectly) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[153,1536] parameter(0)
  p1 = s8[153,128] parameter(1)
  p2 = s8[153,256] parameter(2)
  cat = s8[153,1920] concatenate(p0, p1, p2), dimensions={1}
  cvt = bf16[153,1920] convert(cat)
  p3 = bf16[16,153] parameter(3)
  ROOT d = bf16[16,1920] dot(p3, cvt),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Parameter(), m::Parameter()))));
  const HloComputation* computation =
      module->entry_computation()->root_instruction()->called_computations()[0];
  TF_ASSERT_OK_AND_ASSIGN(const auto analysis,
                          TritonFusionAnalysis::Execute(*computation));

  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(1), 0),
              ElementsAre(FieldsAre(/*stride=*/1536, /*count=*/153,
                                    /*slice_start=*/0, /*sliced_count=*/153,
                                    /*subfragments=*/ElementsAre(153))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(1), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/1536,
                                    /*slice_start=*/0, /*sliced_count=*/1536,
                                    /*subfragments=*/ElementsAre(1536))));

  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(2), 0),
              ElementsAre(FieldsAre(/*stride=*/128, /*count=*/153,
                                    /*slice_start=*/0, /*sliced_count=*/153,
                                    /*subfragments=*/ElementsAre(153))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(2), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/128,
                                    /*slice_start=*/-1536, /*sliced_count=*/128,
                                    /*subfragments=*/ElementsAre(128))));

  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(3), 0),
              ElementsAre(FieldsAre(/*stride=*/256, /*count=*/153,
                                    /*slice_start=*/0, /*sliced_count=*/153,
                                    /*subfragments=*/ElementsAre(153))));
  EXPECT_THAT(*analysis.IterSpec(TritonFusionAnalysis::Scope::RHS,
                                 computation->parameter_instruction(3), 1),
              ElementsAre(FieldsAre(/*stride=*/1, /*count=*/256,
                                    /*slice_start=*/-1536 - 128,
                                    /*sliced_count=*/256,
                                    /*subfragments=*/ElementsAre(256))));
}

TEST_F(GemmFusionLevel2Test, IndivisibleConcatenationIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[124,1024] parameter(0)
  p1 = s8[124,1001] parameter(1)
  cat = s8[124,2025] concatenate(p0, p1), dimensions={1}
  cvt = f16[124,2025] convert(cat)
  p2 = f16[123,124] parameter(2)
  ROOT d = f16[2025,123] dot(cvt, p2),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Parameter()))));
}

TEST_F(GemmFusionLevel2Test, ConcatenationOfContractingIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[124,1024] parameter(0)
  p1 = s8[124,1024] parameter(1)
  cat = s8[124,2048] concatenate(p0, p1), dimensions={1}
  cvt = f16[124,2048] convert(cat)
  p2 = f16[123,2048] parameter(2)
  ROOT d = f16[124,123] dot(cvt, p2),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Parameter()))));
}

TEST_F(GemmFusionLevel2Test, ConcatenationOfBatchIsNotFused) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[124,1024,50] parameter(0)
  p1 = s8[124,1024,50] parameter(1)
  cat = s8[124,2048,50] concatenate(p0, p1), dimensions={1}
  cvt = f16[124,2048,50] convert(cat)
  p2 = f16[123,2048,50] parameter(2)
  ROOT d = f16[2048,124,123] dot(cvt, p2),
    lhs_batch_dims={1}, rhs_batch_dims={1},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})"));
  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Concatenate(), m::Parameter()))));
}

TEST_F(GemmFusionLevel2Test,
       DifferentConcatenationOfSameParametersIsFusedViaNodeDuplication) {
  // It means that the same input is passed to the fusion multiple times and
  // it's read differently for each.
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
e {
  p0 = s8[128,2] parameter(0)
  p1 = s8[128,2] parameter(1)
  cat0 = s8[256,2] concatenate(p0, p1), dimensions={0}
  cvt0 = f16[256,2] convert(cat0)
  cat1 = s8[256,2] concatenate(p1, p0), dimensions={0}
  n1 = s8[256,2] negate(cat1)
  cvt1 = f16[256,2] convert(n1)
  a = f16[256,2] add(cvt1, cvt0)
  p2 = f16[2,18] parameter(2)
  ROOT d = f16[18,256] dot(p2, a),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})"));

  EXPECT_TRUE(GemmFusion(se::CudaComputeCapability{
                             se::CudaComputeCapability::AMPERE, 0})
                  .Run(module.get())
                  .value());
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch((m::Fusion(m::Parameter(), m::Parameter(), m::Parameter(),
                            m::Parameter(), m::Parameter()))));
}

TEST_F(GemmFusionTest, CopiesDotMetadataToFusionOp) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f16[2,18] parameter(0)
  p1 = f16[256,2] parameter(1)
  ROOT d = f16[18,256] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}, metadata={op_name="foo"}
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_EQ(
      module->entry_computation()->root_instruction()->metadata().op_name(),
      "foo");
}

TEST_F(GemmFusionTest, FusesBroadcastOfScalarEpilogues) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m
ENTRY e {
  p0 = f16[2,18] parameter(0)
  p1 = f16[256,2] parameter(1)
  d = f16[18,256] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
  p2 = f16[1] parameter(2)
  p3 = f16[1] parameter(3)
  m0 = f16[1] multiply(f16[1] p2, f16[1] p3)
  bc = f16[] bitcast(m0)
  b = f16[18,256] broadcast(f16[] bc)
  ROOT m = f16[18,256] multiply(d, b)
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::Parameter(), m::Parameter()))));
}

TEST_F(GemmFusionTest, BroadcastsOfParametersAreFusedAsEpilogueInputs) {
  auto module = ParseAndReturnVerifiedModule(R"(
e {
  p0 = f16[4,55] parameter(0)
  p1 = f16[123,55] parameter(1)
  d = f16[4,123] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  p2 = (f16[123,1], f16[456]) parameter(2)
  g = get-tuple-element(p2), index=0
  t = f16[123] bitcast(g)
  b = f16[4,123] broadcast(t), dimensions={1}
  m = f16[4,123] multiply(d, b)
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch((m::Fusion(m::Parameter(), m::Parameter(),
                                    m::GetTupleElement()))));
}

// A test fixture class for testing the threshold for small matrices.
class SmallDotGemmFusionTest : public GemmFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GemmFusionTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(100);
    return debug_options;
  }
};

TEST_F(SmallDotGemmFusionTest, SkipSmallMatrixMultiplicationRewrite) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f16[2,10] parameter(0)
  p1 = f16[10,2] parameter(1)
  ROOT d = f16[10,10] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})")
                    .value();

  EXPECT_FALSE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
; CHECK-LABEL: ENTRY %e ({{.*}}: f16[2,10], {{.*}}: f16[10,2]) -> f16[10,10] {
; CHECK-NEXT: [[P0:%[^ ]+]] = f16[2,10]{1,0} parameter(0)
; CHECK-NEXT: [[P1:%[^ ]+]] = f16[10,2]{1,0} parameter(1)
; CHECK:      ROOT {{.*}} = f16[10,10]{1,0} dot(f16[2,10]{1,0} [[P0]], f16[10,2]{1,0} [[P1]])
})");
}

TEST_F(SmallDotGemmFusionTest, LargeMatrixMultiplicationIsRewritten) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule m

ENTRY e {
  p0 = f16[2,18] parameter(0)
  p1 = f16[50,2] parameter(1)
  ROOT d = f16[18,50] dot(p0, p1),
    lhs_contracting_dims={0}, rhs_contracting_dims={1}
})")
                    .value();

  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
; CHECK-LABEL: ENTRY %e ({{.*}}: f16[2,18], {{.*}}: f16[50,2]) -> f16[18,50] {
; CHECK-NEXT: [[P0:%[^ ]+]] = f16[2,18]{1,0} parameter(0)
; CHECK-NEXT: [[P1:%[^ ]+]] = f16[50,2]{1,0} parameter(1)
; CHECK:      ROOT {{.*}} = f16[18,50]{1,0}
; CHECK:        fusion(f16[2,18]{1,0} [[P0]], f16[50,2]{1,0} [[P1]]),
; CHECK:        kind=kCustom
; CHECK:        __triton_gemm
})");
}

class SparseDotTest : public GemmFusionTest {};

TEST_F(SparseDotTest, DotWithSparseLhsOperandIsRewritten) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test
ENTRY main {
  lhs = f16[2,16] parameter(0)
  rhs = f16[32,2] parameter(1)
  meta = u16[2,2] parameter(2)
  ROOT dot = f32[2,2] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@2:4
})")
                    .value();
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  MatchHloModule(*module, R"(
; CHECK-LABEL: ENTRY %main ({{.*}}: f16[2,16], {{.*}}: f16[32,2], {{.*}}: u16[2,2]) -> f32[2,2] {
; CHECK-NEXT: [[P0:%[^ ]+]] = f16[2,16]{1,0} parameter(0)
; CHECK-NEXT: [[P1:%[^ ]+]] = f16[32,2]{1,0} parameter(1)
; CHECK-NEXT: [[META:%[^ ]+]] = u16[2,2]{1,0} parameter(2)
; CHECK:      ROOT {{.*}} = f32[2,2]{1,0}
; CHECK-SAME:   fusion(f16[2,16]{1,0} [[P0]], f16[32,2]{1,0} [[P1]], u16[2,2]{1,0} [[META]]),
; CHECK-SAME:   kind=kCustom
; CHECK-SAME:   __triton_gemm
})");
}

TEST_F(SparseDotTest, DotWithSparseRhsOperandIsNotSupported) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test
ENTRY main {
  lhs = f16[2,32] parameter(0)
  rhs = f16[16,2] parameter(1)
  meta = u16[2,2] parameter(2)
  ROOT dot = f32[2,2] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=R.0@2:4
})")
                    .value();
  auto result = GemmFusion(gpu_version_).Run(module.get());
  EXPECT_FALSE(result.ok());
}

TEST_F(SparseDotTest, UnsupportedSparsityType) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test
ENTRY main {
  lhs = f16[2,8] parameter(0)
  rhs = f16[32,2] parameter(1)
  meta = u16[2,1] parameter(2)
  ROOT dot = f32[2,2] dot(lhs, rhs, meta),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}, sparsity=L.1@1:4
})")
                    .value();
  auto result = GemmFusion(gpu_version_).Run(module.get());
  EXPECT_FALSE(result.ok());
}

TEST_F(SmallDotGemmFusionTest, Int4DotIsRewritten) {
  constexpr auto kInt4Dot = R"(
    ENTRY e {
      p0 = s8[16,16] parameter(0)
      p1 = s4[16,16] parameter(1)
      p1c = bf16[16,16] convert(p1)
      ROOT dot = bf16[16,16] dot(p0, p1c),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kInt4Dot));
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
}

TEST_F(SmallDotGemmFusionTest, Int4ConcatPlusConvertIsRewritten) {
  const std::string kInt4Dot = R"(
    ENTRY main {
      lhs1 = s4[4,1024]{1,0} parameter(0)
      lhs2 = s4[4,1024]{1,0} parameter(1)
      rhs = bf16[1024,4]{1,0} parameter(2)
      lhs_concat = s4[8,1024]{1,0} concatenate(lhs1, lhs2), dimensions={0}
      lhs_converted = bf16[8,1024]{1,0} convert(lhs_concat)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_converted, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kInt4Dot));
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());

  // Check that the fusion is present and that the lhs is not converted.
  MatchHloModule(*module, R"(
CHECK: gemm_fusion_dot_computation
CHECK:  %parameter_0 = s4[8,1024]{1,0} parameter(0)
CHECK: ENTRY
CHECK-DAG: ROOT {{.*}} = bf16[8,4]{1,0} fusion(s4[8,1024]{1,0} %lhs_concat, bf16[1024,4]{1,0} %rhs)
})");
}

TEST_F(SmallDotGemmFusionTest, Int4ConvertPlusNegateIsRewritten) {
  const std::string kInt4Dot = R"(
    ENTRY main {
      lhs = s4[8,1024]{1,0} parameter(0)
      rhs = f32[1024,4]{1,0} parameter(1)
      lhs_converted = f32[8,1024]{1,0} convert(lhs)
      lhs_negated = f32[8,1024]{1,0} negate(lhs_converted)
      ROOT dot = f32[8,4]{1,0} dot(lhs_negated, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kInt4Dot));
  EXPECT_TRUE(GemmFusion(gpu_version_).Run(module.get()).value());
  // Check that the fusion is present and that convert and negation is fused in
  // it.
  MatchHloModule(*module, R"(
CHECK: gemm_fusion_dot_computation
CHECK:  %parameter_0 = s4[8,1024]{1,0} parameter(0)
CHECK: ENTRY
CHECK-DAG: ROOT {{.*}} = f32[8,4]{1,0} fusion(s4[8,1024]{1,0} %lhs, f32[1024,4]{1,0} %rhs)
})");
}

TEST_F(SmallDotGemmFusionTest, Int4WithMinorBatchDimIsNotRewritten) {
  const std::string kInt4Dot = R"(
    ENTRY main {
      lhs = s4[8,1024,16]{2,1,0} parameter(0)
      lhs_converted = bf16[8,1024,16]{2,1,0} convert(lhs)
      rhs = bf16[16,1024,64]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,64]{2,1,0} dot(lhs_converted, rhs),
        lhs_batch_dims={2},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kInt4Dot));
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          GemmFusion(gpu_version_).Run(module.get()));
  EXPECT_FALSE(result);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
