/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/triton_emitter_constraints.h"

#include <memory>
#include <optional>
#include <utility>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/instruction_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::tsl::testing::IsOkAndHolds;

class TritonEmitterConstraintsTest : public HloTestBase {
 public:
  std::optional<SymbolicTileAnalysis> TryAnalyzeModule(
      HloModule* module, bool with_triton_emitter_specific_constraints = true) {
    EmitterSpecificConstraintsBuilder constraints_builder = nullptr;

    if (with_triton_emitter_specific_constraints) {
      constraints_builder =
          TritonEmitterConstraints::GetBuilder(device_description_);
    }

    SymbolicTileAnalysisOrError analysis_or_error =
        SymbolicTileAnalysis::AnalyzeComputation(
            *module->entry_computation()
                 ->root_instruction()
                 ->fused_instructions_computation(),
            &mlir_context_, constraints_builder);

    if (std::holds_alternative<SymbolicTileAnalysis>(analysis_or_error)) {
      return std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));
    }
    VLOG(1) << "Cannot analyze module: "
            << std::get<FusionDecision>(analysis_or_error).Explain();
    return std::nullopt;
  }

  mlir::MLIRContext mlir_context_;
  se::DeviceDescription device_description_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

TEST_F(TritonEmitterConstraintsTest, TooBigTileSizesConstraintIsEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[8192,50304] parameter(0)
  constant = f32[] constant(-inf)
  reduce = f32[8192] reduce(param_0, constant), dimensions={1}, to_apply=max_computation
  broadcast = f32[8192,50304] broadcast(reduce), dimensions={0}
  ROOT subtract = f32[8192,50304] subtract(param_0, broadcast)
}

ENTRY entry_computation {
  param_0 = f32[8192,50304] parameter(0)
  ROOT fusion = f32[8192,50304] fusion(param_0), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  // The biggest tile in the program has 8 * 65536 = 524288 elements.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({8, 128}),
              IsOkAndHolds(true));

  // The biggest tile in the program is 18 * 50304 = 905472 elements which is
  // smaller than the limit of 1048576, but since Triton requires all tile sizes
  // to be a power of 2, the actual tile will be 32 * 65536 = 2097152 elements.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({18, 50304}),
              IsOkAndHolds(false));

  // Because of reduce, we need to load full rows from param_0 and the load tile
  // will be 1024 * 65536 = 67108864 elements, that is larger than the limit of
  // 1048576.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({1024, 1}),
              IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest, TooManyBlocksConstraintIsEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

max_computation {
  param_0 = f32[] parameter(0)
  param_1 = f32[] parameter(1)
  ROOT maximum = f32[] maximum(param_0, param_1)
}

fused_computation {
  param_0 = f32[65536,65536] parameter(0)
  ROOT log = f32[65536,65536] log(param_0)
}

ENTRY entry_computation {
  param_0 = f32[65536,65536] parameter(0)
  ROOT fusion = f32[65536,65536] fusion(param_0), kind=kCustom, calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
}
)"));

  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  // This tiling will require (65536 * 65536) / (128 * 128) = 262144 blocks.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({128, 128}),
              IsOkAndHolds(true));

  // This would require to run 65538 * 65538 = 4294967296 blocks that is larger
  // than the hardware limit of 2^32 - 1.
  EXPECT_THAT(analysis->ParametersSatisfyConstraints({1, 1}),
              IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest, CustomReshapeConstraintsAreEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
triton_computation {
  p = s8[36] parameter(0)
  ROOT bitcast = s8[6,6] bitcast(p)
}

ENTRY entry_computation {
  p = s8[36] parameter(0)
  ROOT fusion = s8[6,6] fusion(p), kind=kCustom, calls=triton_computation
})"));

  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);

  ASSERT_TRUE(analysis_without_triton_constraints.has_value());

  // (2, 6) is a theoretically valid tiling for this reshape, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints({2, 6}),
      IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (2, 6) is a theoretically valid tiling for this reshape, but it won't
  // work because of Triton's power of two restriction. Thus, we should reject
  // it here.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({2, 6}),
      IsOkAndHolds(false));

  // However, (1, 6) is valid and should still work.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1, 6}),
      IsOkAndHolds(true));
}

TEST_F(TritonEmitterConstraintsTest,
       ReshapeConstraintsAreNotDerivedForFusionOperands) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
triton_computation {
  p = s8[6,6] parameter(0)
  ROOT add = s8[6,6] add(p, p)
}

ENTRY entry_computation {
  p = s8[36] parameter(0)
  bitcast = s8[6,6] bitcast(p)
  ROOT fusion = s8[6,6] fusion(bitcast),
    kind=kCustom, calls=triton_computation
})"));
  std::optional<SymbolicTileAnalysis> analysis = TryAnalyzeModule(module.get());
  ASSERT_TRUE(analysis.has_value());

  const HloComputation* triton_computation =
      FindComputation(module.get(), "triton_computation");

  std::unique_ptr<EmitterSpecificConstraints> constraints =
      TritonEmitterConstraints::GetBuilder(device_description_)(
          analysis->GetSymbolicTiledHloComputation(),
          *HloFusionAdaptor::ForComputation(triton_computation));
  EXPECT_FALSE(reinterpret_cast<TritonEmitterConstraints*>(constraints.get())
                   ->HasCustomConstraints());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
