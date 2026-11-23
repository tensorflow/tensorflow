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

#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/tiling/tiling_space.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"

namespace xla {
namespace cpu {
namespace {

using ::testing::HasSubstr;

class TiledFusionEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; },
        BufferAssigner::Options{});
  }

  AliasInfo alias_info_;
};

TEST_F(TiledFusionEmitterTest, EvaluatesMultipleCandidates) {
  constexpr absl::string_view kReshapeHlo = R"(
    res_computation {
      p0 = f32[36] parameter(0)
      ROOT reshape = f32[6,6] bitcast(p0)
    }

    ENTRY main {
      p0 = f32[36] parameter(0)
      ROOT wrapped_reshape = f32[6,6] fusion(p0), kind=kLoop, calls=res_computation
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kReshapeHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_experimental_enable_tiling_propagation(true);

  ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();

  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_reshape", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());

  // Assert that this fusion has multiple tiling candidates.
  auto fusion_adaptor = HloFusionAdaptor::ForInstruction(fusion);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<gpu::TilingSpace> tiling_space,
      gpu::TilingSpace::Create(*fusion_adaptor, mlir_context.get()));
  ASSERT_OK_AND_ASSIGN(auto candidates, tiling_space->GetValidTilings());
  EXPECT_GT(candidates.size(), 1);
}

TEST_F(TiledFusionEmitterTest, EmitsGemvDotsWithPortedTileSizes) {
  // Use f64 so vector register count (4 for 256-bit) differs from tiling
  // factor (8), verifying row-major vs col-major layout selection.
  constexpr absl::string_view kRowMajorF64Gemv = R"(
    gemv_comp {
      p0 = f64[64,32]{1,0} parameter(0)
      p1 = f64[32] parameter(1)
      ROOT dot = f64[64] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
    ENTRY main {
      p0 = f64[64,32]{1,0} parameter(0)
      p1 = f64[32] parameter(1)
      ROOT wrapped = f64[64] fusion(p0, p1), kind=kLoop, calls=gemv_comp
    }
  )";
  {
    ASSERT_OK_AND_ASSIGN(auto hlo_module,
                         ParseAndReturnVerifiedModule(kRowMajorF64Gemv));
    ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    auto fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    TiledEmissionResult result =
        EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                              "gemv_row", /*num_work_groups=*/1);
    EXPECT_TRUE(result.tiling_succeeded);
    ASSERT_OK(result.kernel.status());
    EXPECT_THAT(result.kernel->source().ToString(),
                HasSubstr("tensor<8x4xf64>"));
  }

  constexpr absl::string_view kColMajorF64Gemv = R"(
    gemv_comp {
      p0 = f64[64,32]{0,1} parameter(0)
      p1 = f64[32] parameter(1)
      ROOT dot = f64[64] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
    ENTRY main {
      p0 = f64[64,32]{0,1} parameter(0)
      p1 = f64[32] parameter(1)
      ROOT wrapped = f64[64] fusion(p0, p1), kind=kLoop, calls=gemv_comp
    }
  )";
  {
    ASSERT_OK_AND_ASSIGN(auto hlo_module,
                         ParseAndReturnVerifiedModule(kColMajorF64Gemv));
    ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    auto fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    TiledEmissionResult result =
        EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                              "gemv_col", /*num_work_groups=*/1);
    EXPECT_TRUE(result.tiling_succeeded);
    ASSERT_OK(result.kernel.status());
    EXPECT_THAT(result.kernel->source().ToString(),
                HasSubstr("tensor<4x8xf64>"));
  }

  constexpr absl::string_view kLhsVectorF64Gemv = R"(
    gemv_comp {
      p0 = f64[1,32]{1,0} parameter(0)
      p1 = f64[32,64]{1,0} parameter(1)
      ROOT dot = f64[1,64] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
    ENTRY main {
      p0 = f64[1,32]{1,0} parameter(0)
      p1 = f64[32,64]{1,0} parameter(1)
      ROOT wrapped = f64[1,64] fusion(p0, p1), kind=kLoop, calls=gemv_comp
    }
  )";
  {
    ASSERT_OK_AND_ASSIGN(auto hlo_module,
                         ParseAndReturnVerifiedModule(kLhsVectorF64Gemv));
    ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    auto fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    TiledEmissionResult result =
        EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                              "gemv_lhs_vec", /*num_work_groups=*/1);
    EXPECT_TRUE(result.tiling_succeeded);
    ASSERT_OK(result.kernel.status());
    EXPECT_THAT(result.kernel->source().ToString(),
                HasSubstr("tensor<8x4xf64>"));
  }

  constexpr absl::string_view kBatchedF64Gemv = R"(
    gemv_comp {
      p0 = f64[2,64,32]{2,1,0} parameter(0)
      p1 = f64[2,32]{1,0} parameter(1)
      ROOT dot = f64[2,64] dot(p0, p1),
        lhs_batch_dims={0}, rhs_batch_dims={0},
        lhs_contracting_dims={2}, rhs_contracting_dims={1}
    }
    ENTRY main {
      p0 = f64[2,64,32]{2,1,0} parameter(0)
      p1 = f64[2,32]{1,0} parameter(1)
      ROOT wrapped = f64[2,64] fusion(p0, p1), kind=kLoop, calls=gemv_comp
    }
  )";
  {
    ASSERT_OK_AND_ASSIGN(auto hlo_module,
                         ParseAndReturnVerifiedModule(kBatchedF64Gemv));
    ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    auto fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    TiledEmissionResult result =
        EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                              "gemv_batched", /*num_work_groups=*/1);
    EXPECT_TRUE(result.tiling_succeeded);
    ASSERT_OK(result.kernel.status());
    EXPECT_THAT(result.kernel->source().ToString(),
                HasSubstr("tensor<1x8x4xf64>"));
  }
}

TEST_F(TiledFusionEmitterTest, RejectsNonGemvDotFusion) {
  constexpr absl::string_view kNonGemvDot = R"(
    dot_comp {
      p0 = f32[3,3] parameter(0)
      p1 = f32[3,3] parameter(1)
      ROOT dot = f32[3,3] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
    ENTRY main {
      p0 = f32[3,3] parameter(0)
      p1 = f32[3,3] parameter(1)
      ROOT wrapped = f32[3,3] fusion(p0, p1), kind=kLoop, calls=dot_comp
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kNonGemvDot));
  ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "non_gemv", /*num_work_groups=*/1);
  EXPECT_FALSE(result.tiling_succeeded);
}

TEST_F(TiledFusionEmitterTest, RejectsPredGemvDotFusion) {
  constexpr absl::string_view kPredGemv = R"(
    gemv_comp {
      p0 = pred[64,32] parameter(0)
      p1 = pred[32] parameter(1)
      ROOT dot = pred[64] dot(p0, p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
    ENTRY main {
      p0 = pred[64,32] parameter(0)
      p1 = pred[32] parameter(1)
      ROOT wrapped = pred[64] fusion(p0, p1), kind=kLoop, calls=gemv_comp
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto hlo_module,
                       ParseAndReturnVerifiedModule(kPredGemv));
  ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                       RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());
  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "pred_gemv", /*num_work_groups=*/1);
  EXPECT_FALSE(result.tiling_succeeded);
}

TEST_F(TiledFusionEmitterTest, CompilesUnsignedGemvDotFusions) {
  for (absl::string_view type : {"u8", "u16", "u32", "u64"}) {
    std::string hlo = absl::Substitute(
        R"hlo(
          gemv_comp {
            p0 = $0[64,32] parameter(0)
            p1 = $0[32] parameter(1)
            ROOT dot = $0[64] dot(p0, p1),
              lhs_contracting_dims={1}, rhs_contracting_dims={0}
          }
          ENTRY main {
            p0 = $0[64,32] parameter(0)
            p1 = $0[32] parameter(1)
            ROOT wrapped = $0[64] fusion(p0, p1), kind=kLoop, calls=gemv_comp
          }
        )hlo",
        type);
    ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(hlo));
    ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                         RunBufferAssignment(*hlo_module));
    auto fusion = Cast<HloFusionInstruction>(
        hlo_module->entry_computation()->root_instruction());
    auto mlir_context = FusionCompiler::CreateContext();
    TiledEmissionResult result =
        EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                              "unsigned_gemv", /*num_work_groups=*/1);
    ASSERT_TRUE(result.tiling_succeeded);
    ASSERT_OK_AND_ASSIGN(auto kernel_def, std::move(result.kernel));
    FusionCompiler compiler(
        mlir_context.get(),
        FusionCompiler::Options{/*vector_width=*/256, /*verification_level=*/1,
                                /*fast_min_max=*/false});
    EXPECT_OK(compiler.Compile(std::move(kernel_def).TakeSource()));
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
