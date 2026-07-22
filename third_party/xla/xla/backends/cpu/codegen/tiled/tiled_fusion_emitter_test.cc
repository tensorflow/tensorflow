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

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Verifier.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

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
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnVerifiedModule(kReshapeHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_experimental_enable_tiling_propagation(true);

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
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
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<gpu::experimental::TilingSpace> tiling_space,
      gpu::experimental::TilingSpace::Create(*fusion_adaptor,
                                             mlir_context.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto candidates, tiling_space->GetValidTilings());
  EXPECT_GT(candidates.size(), 1);
}

TEST_F(TiledFusionEmitterTest, EmitsMultiOutputFusion) {
  constexpr absl::string_view kHlo = R"(
    multi_out_computation {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      add1 = f32[8,8] add(p0, p1)
      sub1 = f32[8,8] subtract(add1, p1)
      ROOT tuple1 = (f32[8,8], f32[8,8]) tuple(add1, sub1)
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      p1 = f32[8,8] parameter(1)
      ROOT wrapped = (f32[8,8], f32[8,8]) fusion(p0, p1), kind=kLoop, calls=multi_out_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_multi_out", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));

  std::string mlir_str;
  llvm::raw_string_ostream os(mlir_str);
  result.kernel->source().module()->print(os);

  constexpr absl::string_view kExpectedIR = R"(
    CHECK: @wrapped_multi_out
  )";
  TF_ASSERT_OK_AND_ASSIGN(bool matched, RunFileCheck(mlir_str, kExpectedIR));
  EXPECT_TRUE(matched);
}

TEST_F(TiledFusionEmitterTest, EmitsReverseFusion) {
  constexpr absl::string_view kHlo = R"(
    reverse_computation {
      p0 = f32[8,8] parameter(0)
      ROOT rev = f32[8,8] reverse(p0), dimensions={0,1}
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      ROOT wrapped = f32[8,8] fusion(p0), kind=kLoop, calls=reverse_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_reverse", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

TEST_F(TiledFusionEmitterTest, EmitsDynamicUpdateSliceFusion) {
  constexpr absl::string_view kHlo = R"(
    dus_computation {
      p0 = f32[8,8] parameter(0)
      p1 = f32[2,2] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT dus = f32[8,8] dynamic-update-slice(p0, p1, p2, p3)
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      p1 = f32[2,2] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT wrapped = f32[8,8] fusion(p0, p1, p2, p3), kind=kLoop, calls=dus_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_dus", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));

  std::string mlir_str;
  llvm::raw_string_ostream os(mlir_str);
  result.kernel->source().module()->print(os);

  constexpr absl::string_view kExpectedIR = R"(
    CHECK: @wrapped_dus
  )";
  TF_ASSERT_OK_AND_ASSIGN(bool matched, RunFileCheck(mlir_str, kExpectedIR));
  EXPECT_TRUE(matched);
}

TEST_F(TiledFusionEmitterTest, EmitsReduceWindowFusion) {
  constexpr absl::string_view kHlo = R"(
    add_computation {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    rw_computation {
      p0 = f32[8,8] parameter(0)
      c0 = f32[] constant(0)
      ROOT rw = f32[6,6] reduce-window(p0, c0), window={size=3x3}, to_apply=add_computation
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      ROOT wrapped = f32[6,6] fusion(p0), kind=kLoop, calls=rw_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_rw", /*num_work_groups=*/1);

  EXPECT_FALSE(result.tiling_succeeded);
}

TEST_F(TiledFusionEmitterTest, EmitsGatherFusion) {
  constexpr absl::string_view kHlo = R"(
    gather_computation {
      p0 = f32[10,20] parameter(0)
      p1 = s32[3,1] parameter(1)
      ROOT gather = f32[3,1,20] gather(p0, p1),
        offset_dims={1,2},
        collapsed_slice_dims={},
        start_index_map={0},
        index_vector_dim=1,
        slice_sizes={1,20}
    }

    ENTRY main {
      p0 = f32[10,20] parameter(0)
      p1 = s32[3,1] parameter(1)
      ROOT wrapped = f32[3,1,20] fusion(p0, p1), kind=kLoop, calls=gather_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_gather", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

TEST_F(TiledFusionEmitterTest, EmitsSliceFusion) {
  constexpr absl::string_view kHlo = R"(
    slice_computation {
      p0 = f32[8,8] parameter(0)
      ROOT slice = f32[8,8] slice(p0), slice={[0:8], [0:8]}
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      ROOT wrapped = f32[8,8] fusion(p0), kind=kLoop, calls=slice_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_slice", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

TEST_F(TiledFusionEmitterTest, EmitsDynamicSliceFusion) {
  constexpr absl::string_view kHlo = R"(
    ds_computation {
      p0 = f32[8,8] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      ROOT ds = f32[2,2] dynamic-slice(p0, p1, p2), dynamic_slice_sizes={2,2}
    }

    ENTRY main {
      p0 = f32[8,8] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      ROOT wrapped = f32[2,2] fusion(p0, p1, p2), kind=kLoop, calls=ds_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_ds", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

TEST_F(TiledFusionEmitterTest, EmitsSmallDotFusion) {
  constexpr absl::string_view kHlo = R"(
    dot_computation {
      p0 = f32[8,16] parameter(0)
      p1 = f32[16,32] parameter(1)
      ROOT dot = f32[8,32] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      p0 = f32[8,16] parameter(0)
      p1 = f32[16,32] parameter(1)
      ROOT wrapped = f32[8,32] fusion(p0, p1), kind=kLoop, calls=dot_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_dot", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

TEST_F(TiledFusionEmitterTest, EmitsSmallReduceFusion) {
  constexpr absl::string_view kHlo = R"(
    add_computation {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    reduce_computation {
      p0 = f32[8,16] parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[8] reduce(p0, c0), dimensions={1}, to_apply=add_computation
    }

    ENTRY main {
      p0 = f32[8,16] parameter(0)
      ROOT wrapped = f32[8] fusion(p0), kind=kLoop, calls=reduce_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TiledEmissionResult result =
      EmitTiledFusionKernel(*mlir_context, *fusion, buffer_assignment.get(),
                            "wrapped_reduce", /*num_work_groups=*/1);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

TEST_F(TiledFusionEmitterTest,
       EvaluatesCostModeledTileSizeSelectionWithTargetMachineFeatures) {
  constexpr absl::string_view kHlo = R"(
    fused_computation {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      add0 = f32[64,64] add(p0, p1)
      mul0 = f32[64,64] multiply(add0, p1)
      ROOT add1 = f32[64,64] add(add0, mul0)
    }

    ENTRY main {
      p0 = f32[64,64] parameter(0)
      p1 = f32[64,64] parameter(1)
      ROOT wrapped = f32[64,64] fusion(p0, p1), kind=kLoop, calls=fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module, ParseAndReturnVerifiedModule(kHlo));
  auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
  debug_options.set_xla_cpu_experimental_enable_tiling_propagation(true);

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment,
                          RunBufferAssignment(*hlo_module));
  auto fusion = Cast<HloFusionInstruction>(
      hlo_module->entry_computation()->root_instruction());

  auto mlir_context = FusionCompiler::CreateContext();
  TargetMachineFeatures target_machine_features(nullptr);

  TiledEmissionResult result = EmitTiledFusionKernel(
      *mlir_context, *fusion, buffer_assignment.get(), "wrapped_cost_modeled",
      /*num_work_groups=*/1, &target_machine_features);

  EXPECT_TRUE(result.tiling_succeeded);
  EXPECT_OK(result.kernel.status());
  EXPECT_TRUE(mlir::succeeded(mlir::verify(result.kernel->source().module())));
}

}  // namespace
}  // namespace cpu
}  // namespace xla
