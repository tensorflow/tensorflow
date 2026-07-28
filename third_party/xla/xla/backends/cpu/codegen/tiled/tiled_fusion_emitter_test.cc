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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
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
      std::unique_ptr<gpu::experimental::TilingSpace> tiling_space,
      gpu::experimental::TilingSpace::Create(*fusion_adaptor,
                                             mlir_context.get()));
  ASSERT_OK_AND_ASSIGN(auto candidates, tiling_space->GetValidTilings());
  EXPECT_GT(candidates.size(), 1);
}

TEST_F(TiledFusionEmitterTest, CompilesToLlvmWithBothLowerings) {
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

  ASSERT_TRUE(result.tiling_succeeded);
  ASSERT_OK(result.kernel.status());

  ASSERT_OK_AND_ASSIGN(auto kernel_def, std::move(result.kernel));
  MlirKernelSource mlir_source = std::move(kernel_def).TakeSource();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      std::move(mlir_source).TakeModule();

  // Compile with use_new_xtile_lowering = false
  {
    FusionCompiler::Options options;
    options.vector_width = 256;
    options.verification_level = 1;
    options.fast_min_max = true;
    options.use_new_xtile_lowering = false;

    mlir::OwningOpRef<mlir::ModuleOp> cloned_module(
        mlir::cast<mlir::ModuleOp>(module.get()->clone()));

    FusionCompiler compiler(mlir_context.get(), options);
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    EXPECT_OK(compiler.Compile(*llvm_context, *cloned_module).status());
  }

  // Compile with use_new_xtile_lowering = true
  {
    FusionCompiler::Options options;
    options.vector_width = 256;
    options.verification_level = 1;
    options.fast_min_max = true;
    options.use_new_xtile_lowering = true;

    mlir::OwningOpRef<mlir::ModuleOp> cloned_module(
        mlir::cast<mlir::ModuleOp>(module.get()->clone()));

    FusionCompiler compiler(mlir_context.get(), options);
    auto llvm_context = std::make_unique<llvm::LLVMContext>();
    // Currently this is expected to fail because
    // AddXtileToVectorPassesNoBufferization does not lower xtile.extract.
    EXPECT_FALSE(compiler.Compile(*llvm_context, *cloned_module).ok());
  }
}

}  // namespace
}  // namespace cpu
}  // namespace xla
