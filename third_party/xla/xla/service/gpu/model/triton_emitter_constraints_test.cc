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
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

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

TEST_F(TritonEmitterConstraintsTest,
       CustomConcatenateSizeConstraintsAreEnforced) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[8] parameter(2)
  ROOT concatenate = bf16[24] concatenate(p0, p1, p2), dimensions={0}
}

ENTRY main {
  p0 = bf16[8] parameter(0)
  p1 = bf16[8] parameter(1)
  p2 = bf16[8] parameter(2)
  ROOT fusion = bf16[24] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());

  // (16,) is a theoretically valid tiling for this concatenate, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints({16}),
      IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (16,) is a theoretically valid tiling for this concatenate, but it won't
  // work in our lowering for now, because we want to be loading from a single
  // operand at a time, and it doesn't divide each operand's concatenation
  // dimension. We want to reject it here.
  //
  // Note: this is perfectly OK to expand later as our codegen improves to
  // handle this case.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({16}),
      IsOkAndHolds(false));

  // However, (4,) is valid and should still work.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({4}),
      IsOkAndHolds(true));
}

TEST_F(TritonEmitterConstraintsTest,
       ConcatenateConstrainsOffsetToBeZeroAlongConcatenationDimension) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  concatenate = bf16[48] concatenate(p0, p1, p2), dimensions={0}
  ROOT slice = bf16[24] slice(concatenate), slice={[24:48]}
}

ENTRY main {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  ROOT fusion = bf16[24] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());

  // (8,) is a theoretically valid tiling for this concatenate, and one that
  // works for all operands, so SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints({8}),
      IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (8,) is a theoretically valid tiling for this concatenate, but the
  // constraints enforce that the offset along the concatenation dimension be 0.
  // Here, it is 24, so we expect the tiling to be rejected.
  //
  // Note: this is perfectly OK to expand later as our codegen improves to
  // handle this case.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({8}),
      IsOkAndHolds(false));

  // Even the smallest tiling, (1,) should be rejected here. (This is
  // unnecessary in theory, but a sanity check for the implementation).
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1}),
      IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest,
       ConcatenateConstrainsStrideToBeOneAlongConcatenationDimension) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
concatenate {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  concatenate = bf16[48] concatenate(p0, p1, p2), dimensions={0}
  ROOT slice = bf16[24] slice(concatenate), slice={[0:48:2]}
}

ENTRY main {
  p0 = bf16[16] parameter(0)
  p1 = bf16[16] parameter(1)
  p2 = bf16[16] parameter(2)
  ROOT fusion = bf16[24] fusion(p0, p1, p2),
    kind=kCustom, calls=concatenate, backend_config={"fusion_backend_config":{
      "kind":"__triton_nested_gemm_fusion"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());

  // (8,) is a theoretically valid tiling for this concatenate, and one that
  // works for all operands, so SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints({8}),
      IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (8,) is a theoretically valid tiling for this concatenate, but the
  // constraints enforce that the stride along the concatenation dimension be 1.
  // Here, it is 2, so we expect the tiling to be rejected.
  //
  // Note: this is perfectly OK to expand later as our codegen improves to
  // handle this case.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({8}),
      IsOkAndHolds(false));

  // Even the smallest tiling, (1,) should be rejected here. (This is
  // unnecessary in theory, but a sanity check for the implementation).
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1}),
      IsOkAndHolds(false));
}

TEST_F(TritonEmitterConstraintsTest, FusionHasValidTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

fused_computation {
  param_0 = f32[36] parameter(0)
  abs = f32[36] abs(param_0)
  ROOT reshape = f32[6,6] reshape(abs)
}

ENTRY entry_computation {
  param_0 = f32[36] parameter(0)
  ROOT fusion = f32[6,6] fusion(param_0), kind=kCustom,
    calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());

  // (1,3) is a theoretically valid tiling for this fusion, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints({1, 3}),
      IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (1,3) is a theoretically valid tiling for this fusion, but it does not pass
  // the triton specific condition that all tile sizes are either powers of 2,
  // or equal to the dimension size.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1, 3}),
      IsOkAndHolds(false));

  // However if we capture the last dimension fully, it should be valid.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1, 6}),
      IsOkAndHolds(true));

  // Also powers of 2 are valid.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({2, 1}),
      IsOkAndHolds(true));
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1, 8}),
      IsOkAndHolds(true));
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1, 4}),
      IsOkAndHolds(true));
}

TEST_F(TritonEmitterConstraintsTest, MultiOutputFusionHasPowerOfTwoTileSizes) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

add {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_computation {
  param_0 = f32[36] parameter(0)
  abs = f32[36] abs(param_0)
  reshape = f32[3,12] reshape(abs)
  zero = f32[] constant(0)
  reduce = f32[3] reduce(reshape, zero), to_apply=add, dimensions={1}
  ROOT tuple = (f32[3], f32[36]) tuple(reduce, abs)
}

ENTRY entry_computation {
  param_0 = f32[36] parameter(0)
  ROOT fusion = (f32[3], f32[36]) fusion(param_0), kind=kCustom,
    calls=fused_computation, backend_config={"fusion_backend_config":{"kind":"__triton"}}
})"));
  std::optional<SymbolicTileAnalysis> analysis_without_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/false);
  ASSERT_TRUE(analysis_without_triton_constraints.has_value());

  // (1,) is a theoretically valid tiling for this multi-output fusion, so
  // SymbolicTileAnalysis should allow it.
  EXPECT_THAT(
      analysis_without_triton_constraints->ParametersSatisfyConstraints({1}),
      IsOkAndHolds(true));

  std::optional<SymbolicTileAnalysis> analysis_with_triton_constraints =
      TryAnalyzeModule(module.get(),
                       /*with_triton_emitter_specific_constraints=*/true);

  ASSERT_TRUE(analysis_with_triton_constraints.has_value());

  // (1,) is a theoretically valid tiling for this multi-output fusion, but the
  // propagated tile size of (1,12) for the extra output does not pass the
  // condition that all tile sizes are powers of 2. This can result in different
  // paddings for the different roots being used, which can cause problems if
  // buffers are shared.
  EXPECT_THAT(
      analysis_with_triton_constraints->ParametersSatisfyConstraints({1}),
      IsOkAndHolds(false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
