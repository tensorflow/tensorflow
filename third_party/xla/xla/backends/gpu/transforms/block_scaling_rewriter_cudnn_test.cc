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

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "third_party/gpus/cudnn/cudnn_version.h"
#include "xla/backends/gpu/transforms/block_scaling_rewriter.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"

namespace xla::gpu {
namespace {

using BlockScalingRewriterCudnnTest = HloInterpreterReferenceMixin<HloTestBase>;

const se::dnn::VersionInfo kCudnnDisabled;
const se::dnn::VersionInfo kCudnnVersion(CUDNN_VERSION / 10000,
                                         CUDNN_VERSION % 10000 / 100,
                                         CUDNN_VERSION % 100);

TEST_F(BlockScalingRewriterCudnnTest, Mxfp8) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[256,256] parameter(0)
  %rhs = f8e4m3fn[256,256] parameter(1)
  %lhs_scale = f8e8m0fnu[256,8] parameter(2)
  %rhs_scale = f8e8m0fnu[256,8] parameter(3)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

TEST_F(BlockScalingRewriterCudnnTest, Mxfp8_MixedTypes) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,128,224] parameter(0)
  %rhs = f8e5m2[4,120,224] parameter(1)
  %lhs_scale = f8e8m0fnu[4,128,7] parameter(2)
  %rhs_scale = f8e8m0fnu[4,120,7] parameter(3)
  ROOT %result = f32[4,128,120] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

TEST_F(BlockScalingRewriterCudnnTest, Mxfp8_PaddedScales) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[250,224] parameter(0)
  %rhs = f8e4m3fn[250,224] parameter(1)
  %lhs_scale = f8e8m0fnu[250,7] parameter(2)
  %rhs_scale = f8e8m0fnu[250,7] parameter(3)
  %min_scale = f8e8m0fnu[] constant(5.87747e-39)
  %lhs_scale_padded = f8e8m0fnu[256,8] pad(%lhs_scale, %min_scale), padding=0_6x0_1
  %rhs_scale_padded = f8e8m0fnu[256,8] pad(%rhs_scale, %min_scale), padding=0_6x0_1
  ROOT %result = f32[250,250] custom-call(%lhs, %rhs, %lhs_scale_padded, %rhs_scale_padded),
      custom_call_target="__op$block_scaled_dot",
      backend_config={"block_scaled_dot_backend_config":{block_size:32}}
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

// Scale E2M1FN inputs, as otherwise they become all zeros for the random
// distribution produced by the test due to low type precision.
// Use positive block scale values, as Blackwell MMA discards the sign bit on
// the scale tensor.
TEST_F(BlockScalingRewriterCudnnTest, Nvfp4) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %mult_scalar = f16[] constant(6)
  %mult = f16[256,256] broadcast(%mult_scalar), dimensions={}
  %p0 = f16[256,256] parameter(0)
  %p1 = f16[256,256] parameter(1)
  %lhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p0, %mult))
  %rhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p1, %mult))
  %p2 = f8e4m3fn[256,16] parameter(2)
  %p3 = f8e4m3fn[256,16] parameter(3)
  %lhs_scale = f8e4m3fn[256,16] abs(%p2)
  %rhs_scale = f8e4m3fn[256,16] abs(%p3)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

TEST_F(BlockScalingRewriterCudnnTest, Nvfp4_GlobalScale) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %mult_scalar = f16[] constant(6)
  %mult = f16[256,256] broadcast(%mult_scalar), dimensions={}
  %p0 = f16[256,256] parameter(0)
  %p1 = f16[256,256] parameter(1)
  %lhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p0, %mult))
  %rhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p1, %mult))
  %p2 = f8e4m3fn[256,16] parameter(2)
  %p3 = f8e4m3fn[256,16] parameter(3)
  %lhs_scale = f8e4m3fn[256,16] abs(%p2)
  %rhs_scale = f8e4m3fn[256,16] abs(%p3)
  %global_scale = f32[] parameter(4)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale, %global_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

TEST_F(BlockScalingRewriterCudnnTest, Mxfp8_E5M2) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %lhs = f8e5m2[256,256] parameter(0)
  %rhs = f8e5m2[256,256] parameter(1)
  %lhs_scale = f8e8m0fnu[256,8] parameter(2)
  %rhs_scale = f8e8m0fnu[256,8] parameter(3)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

TEST_F(BlockScalingRewriterCudnnTest, Mxfp4) {
  constexpr absl::string_view hlo_string = R"(
HloModule test

ENTRY main {
  %mult_scalar = f16[] constant(6)
  %mult = f16[256,256] broadcast(%mult_scalar), dimensions={}
  %p0 = f16[256,256] parameter(0)
  %p1 = f16[256,256] parameter(1)
  %lhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p0, %mult))
  %rhs = f4e2m1fn[256,256] convert(f16[256,256] multiply(%p1, %mult))
  %lhs_scale = f8e8m0fnu[256,8] parameter(2)
  %rhs_scale = f8e8m0fnu[256,8] parameter(3)
  ROOT %result = f32[256,256] custom-call(%lhs, %rhs, %lhs_scale, %rhs_scale),
      custom_call_target="__op$block_scaled_dot"
})";

  EXPECT_TRUE(RunAndCompare(
      hlo_string, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
      /*reference_preprocessor=*/
      [](HloModule* reference_module) {
        BlockScalingRewriter pass(kCudnnDisabled);
        EXPECT_THAT(RunHloPass(&pass, reference_module),
                    absl_testing::IsOkAndHolds(true));
      },
      /*test_preprocessor=*/
      [](HloModule* test_module) {
        BlockScalingRewriter pass(kCudnnVersion);
        EXPECT_THAT(RunHloPass(&pass, test_module),
                    absl_testing::IsOkAndHolds(true));
      }));

  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnDisabled),
                            "CHECK-NOT: __cudnn$blockScaledDot");
  RunAndFilecheckHloRewrite(hlo_string, BlockScalingRewriter(kCudnnVersion),
                            "CHECK: __cudnn$blockScaledDot");
}

class CudnnScaledDotRewriterCudnnTest : public BlockScalingRewriterCudnnTest {
 protected:
  // Runs CudnnScaledDotRewriter on `hlo` (with layouts already assigned) and
  // compares the result against the interpreter.
  void RunAndCompareRewriter(absl::string_view hlo) {
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        hlo, ErrorSpec(/*aabs=*/1e-4, /*arel=*/1e-5),
        /*reference_preprocessor=*/nullptr,
        /*test_preprocessor=*/[](HloModule* test_module) {
          CudnnScaledDotRewriter pass;
          EXPECT_THAT(RunHloPass(&pass, test_module),
                      absl_testing::IsOkAndHolds(true));
        }));
  }
};

TEST_F(CudnnScaledDotRewriterCudnnTest, PaddedScales3dNN) {
  RunAndCompareRewriter(R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[4,256,224]{2,1,0} parameter(0)
  %rhs = f8e5m2[4,224,128]{1,2,0} parameter(1)
  %lhs_scale = f8e8m0fnu[4,256,7]{2,1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[4,7,128]{1,2,0} parameter(3)
  ROOT %result = f32[4,256,128]{2,1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={0}, rhs_batch_dims={0},
      lhs_contracting_dims={2}, rhs_contracting_dims={1}
})");
}

TEST_F(CudnnScaledDotRewriterCudnnTest, PaddedContracting2dTN) {
  RunAndCompareRewriter(R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[64,128]{0,1} parameter(0)
  %rhs = f8e4m3fn[64,256]{0,1} parameter(1)
  %lhs_scale = f8e8m0fnu[2,128]{0,1} parameter(2)
  %rhs_scale = f8e8m0fnu[2,256]{0,1} parameter(3)
  ROOT %result = f32[128,256]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={0}, rhs_contracting_dims={0}
})");
}

TEST_F(CudnnScaledDotRewriterCudnnTest, Aligned2dNN) {
  RunAndCompareRewriter(R"(
HloModule test

ENTRY main {
  %lhs = f8e4m3fn[128,256]{1,0} parameter(0)
  %rhs = f8e4m3fn[256,128]{0,1} parameter(1)
  %lhs_scale = f8e8m0fnu[128,8]{1,0} parameter(2)
  %rhs_scale = f8e8m0fnu[8,128]{0,1} parameter(3)
  ROOT %result = bf16[128,128]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
})");
}

TEST_F(CudnnScaledDotRewriterCudnnTest, Nvfp4K32) {
  RunAndCompareRewriter(R"(
HloModule test

ENTRY main {
  %mult_scalar = f16[] constant(6)
  %mult_lhs = f16[128,4,32]{2,0,1} broadcast(%mult_scalar), dimensions={}
  %mult_rhs = f16[32,256,4]{0,1,2} broadcast(%mult_scalar), dimensions={}
  %p0 = f16[128,4,32]{2,0,1} parameter(0)
  %p1 = f16[32,256,4]{0,1,2} parameter(1)
  %lhs = f4e2m1fn[128,4,32]{2,0,1} convert(f16[128,4,32]{2,0,1} multiply(%p0, %mult_lhs))
  %rhs = f4e2m1fn[32,256,4]{0,1,2} convert(f16[32,256,4]{0,1,2} multiply(%p1, %mult_rhs))
  %p2 = f8e4m3fn[128,4,2]{2,0,1} parameter(2)
  %p3 = f8e4m3fn[2,256,4]{0,1,2} parameter(3)
  %lhs_scale = f8e4m3fn[128,4,2]{2,0,1} abs(%p2)
  %rhs_scale = f8e4m3fn[2,256,4]{0,1,2} abs(%p3)
  ROOT %result = bf16[4,128,256]{2,1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_batch_dims={1}, rhs_batch_dims={2},
      lhs_contracting_dims={2}, rhs_contracting_dims={0}
})");
}

struct MxType {
  absl::string_view name;
  absl::string_view input;
  absl::string_view scale;
  int64_t block_size;
};

constexpr MxType kMxTypes[] = {
    {"mxfp8", "f8e4m3fn", "f8e8m0fnu", 32},
    {"mxfp4", "f4e2m1fn", "f8e8m0fnu", 32},
    {"nvfp4", "f4e2m1fn", "f8e4m3fn", 16},
};

// {mx type, output type, column-major output, batched, M, N, K}.
using SweepParam = std::tuple<MxType, absl::string_view, bool, bool, int64_t,
                              int64_t, int64_t>;

// Checks that every scaled dot accepted by IsSupportedPostLayout compiles with
// cuDNN on the current device after CudnnScaledDotRewriter, i.e. that the
// rewriter never emits a fusion cuDNN rejects (there is no fallback for it).
class CudnnScaledDotSupportSweepTest
    : public BlockScalingRewriterCudnnTest,
      public ::testing::WithParamInterface<SweepParam> {
 protected:
  static std::string Hlo(const SweepParam& param) {
    auto [type, out_type, col_major, batched, m, n, k] = param;
    std::string batch = batched ? "2," : "";
    absl::string_view in_layout = batched ? "{2,1,0}" : "{1,0}";
    absl::string_view out_layout = batched ? (col_major ? "{1,2,0}" : "{2,1,0}")
                                           : (col_major ? "{0,1}" : "{1,0}");
    std::string dnums = absl::Substitute(
        "$0lhs_contracting_dims={$1}, rhs_contracting_dims={$1}",
        batched ? "lhs_batch_dims={0}, rhs_batch_dims={0}, " : "",
        batched ? 2 : 1);
    return absl::Substitute(
               R"(
HloModule test
ENTRY main {
  %lhs = $0[$2$3,$5]$8 parameter(0)
  %rhs = $0[$2$4,$5]$8 parameter(1)
  %lhs_scale = $1[$2$3,$6]$8 parameter(2)
  %rhs_scale = $1[$2$4,$6]$8 parameter(3)
  ROOT %result = $7[$2$3,$4]$9 scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      )",
               type.input, type.scale, batch, m, n, k, k / type.block_size,
               out_type, in_layout, out_layout) +
           dnums + "\n}";
  }
};

TEST_P(CudnnScaledDotSupportSweepTest, SupportedImpliesCompiles) {
  const MxType& type = std::get<0>(GetParam());
  int64_t k = std::get<6>(GetParam());
  if (k % type.block_size != 0) {
    GTEST_SKIP() << "K is not a multiple of the block size.";
  }
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(Hlo(GetParam())));
  if (!CudnnScaledDotHelper::IsSupportedPostLayout(
          Cast<HloScaledDotInstruction>(
              module->entry_computation()->root_instruction()))) {
    GTEST_SKIP() << "Rejected by IsSupportedPostLayout.";
  }
  CudnnScaledDotRewriter pass;
  ASSERT_THAT(RunHloPass(&pass, module.get()),
              absl_testing::IsOkAndHolds(true));
  EXPECT_OK(CreateExecutable(std::move(module), /*run_hlo_passes=*/false));
}

std::string SweepParamToString(
    const ::testing::TestParamInfo<SweepParam>& info) {
  auto [type, out_type, col_major, batched, m, n, k] = info.param;
  return absl::StrCat(type.name, "_", out_type, col_major ? "_colmajor" : "",
                      batched ? "_batched" : "", "_m", m, "_n", n, "_k", k);
}

// Covers f32 and bf16 outputs, and K values that are (un)aligned to the 16-byte
// TMA requirement for FP4.
INSTANTIATE_TEST_SUITE_P(
    Sweep, CudnnScaledDotSupportSweepTest,
    ::testing::Combine(::testing::ValuesIn(kMxTypes),
                       ::testing::Values("f32", "bf16"), ::testing::Bool(),
                       ::testing::Bool(), ::testing::Values(128, 256),
                       ::testing::Values(128, 384),
                       ::testing::Values(16, 32, 64, 96, 128, 256, 512)),
    SweepParamToString);

// The tests below pin the cuDNN constraints that IsSupportedPostLayout and
// CudnnScaledDotRewriter work around. If cuDNN starts accepting these fusions,
// the corresponding restriction can be relaxed.
using CudnnScaledDotConstraintTest = BlockScalingRewriterCudnnTest;

TEST_F(CudnnScaledDotConstraintTest, CudnnRejectsTmaUnalignedK) {
  // K=16 of 4-bit F4E2M1FN is 8 bytes, below the 16-byte TMA alignment.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(R"(
HloModule test
fusion {
  %lhs = f4e2m1fn[128,16]{1,0} parameter(0)
  %rhs = f4e2m1fn[128,16]{1,0} parameter(1)
  %lhs_scale = f8e4m3fn[128,4]{1,0} parameter(2)
  %rhs_scale = f8e4m3fn[128,4]{1,0} parameter(3)
  ROOT %result = f32[128,128]{1,0} scaled-dot(%lhs, %rhs, %lhs_scale, %rhs_scale),
      lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
ENTRY main {
  %lhs = f4e2m1fn[128,16]{1,0} parameter(0)
  %rhs = f4e2m1fn[128,16]{1,0} parameter(1)
  %lhs_scale = f8e4m3fn[128,4]{1,0} parameter(2)
  %rhs_scale = f8e4m3fn[128,4]{1,0} parameter(3)
  ROOT %result = f32[128,128]{1,0} fusion(%lhs, %rhs, %lhs_scale, %rhs_scale),
      kind=kCustom, calls=fusion,
      backend_config={"fusion_backend_config":{kind:"__cudnn$fusion"}}
})"));
  EXPECT_FALSE(
      CreateExecutable(std::move(module), /*run_hlo_passes=*/false).ok());
}

}  // namespace
}  // namespace xla::gpu
