/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/service/gpu/fusions/reduction_base.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::SizeIs;

class ReductionTest : public HloTestBase {
 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context_;
};

class FakeReductionFusion : public ReductionFusionBase<KernelFusionInterface> {
  using ReductionFusionBase::ReductionFusionBase;
  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext&, const HloFusionInstruction&) const override {
    return absl::UnimplementedError("Unimplemented");
  }
};

class FakeMlirReductionFusion
    : public ReductionFusionBase<KernelFusionInterface, true> {
  using ReductionFusionBase::ReductionFusionBase;
  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext&, const HloFusionInstruction&) const override {
    return absl::UnimplementedError("Unimplemented");
  }
};

std::unique_ptr<FakeReductionFusion> GetReductionFusion(
    const HloFusionAnalysis& analysis) {
  return std::make_unique<FakeReductionFusion>(analysis);
}

TEST_F(ReductionTest, ThreadIndexingRowReduction) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %input = f32[100,64,512] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
    }

    ENTRY entry {
      %input = f32[100,64,512] parameter(0)
      ROOT %fusion = f32[100,64] fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (
          (d3 * 8 + d0 floordiv 32) floordiv 64,
          (d3 * 8 + d0 floordiv 32) mod 64,
          (d0 mod 32 + s2 * 32) * 2 + s3
        )
        domain:
        d0 in [0, 255]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 799]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 0]
        s2 in [0, 7]
        s3 in [0, 1]
        d0 mod 32 + s2 * 32 in [0, 255]
        d3 * 8 + d0 floordiv 32 in [0, 6399]
      )"));
  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5) -> (
          (d3 * 8 + d0 floordiv 32) floordiv 64,
          (d3 * 8 + d0 floordiv 32) mod 64
        )
        domain:
        d0 in [0, 255]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 799]
        d4 in [0, 0]
        d5 in [0, 0]
        d0 mod 32 in [0, 0]
        d3 * 8 + d0 floordiv 32 in [0, 6399]
      )"));
}

TEST_F(ReductionTest, ThreadIndexingMultiRowReduction) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %input = f32[100,64,4] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
    }

    ENTRY entry {
      %input = f32[100,64,4] parameter(0)
      ROOT %fusion = f32[100,64] fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 + (d0 floordiv 4) floordiv 64,
          (d0 floordiv 4) mod 64,
          d0 mod 4
        )
        domain:
        d0 in [0, 255]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 99]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 0]
        s2 in [0, 0]
        d0 mod 4 in [0, 3]
        d3 * 64 + d0 floordiv 4 in [0, 6399]
      )"));
  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5) -> (
          d3 + (d0 floordiv 4) floordiv 64,
          (d0 floordiv 4) mod 64
        )
        domain:
        d0 in [0, 255]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 99]
        d4 in [0, 0]
        d5 in [0, 0]
        d0 mod 4 in [0, 0]
        d3 * 64 + d0 floordiv 4 in [0, 6399]
      )"));
}

TEST_F(ReductionTest, ThreadIndexingColumnReduction) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %input = f32[100,64,32] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,32] reduce(%input, %c0), dimensions={1}, to_apply=add
    }

    ENTRY entry {
      %input = f32[100,64,32] parameter(0)
      ROOT %fusion = f32[100,32] fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3,
          d0 floordiv 32 + s1 * 32,
          d0 mod 32
        )
        domain:
        d0 in [0, 1023] d1 in [0, 0] d2 in [0, 0]
        d3 in [0, 99] d4 in [0, 0] d5 in [0, 0]
        s0 in [0, 0] s1 in [0, 127] s2 in [0, 0]
        d0 floordiv 32 + s1 * 32 in [0, 63]
        d0 mod 32 in [0, 31]
      )"));
  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5) -> (
          d3,
          d0 floordiv 32
        )
        domain:
        d0 in [0, 1023] d1 in [0, 0] d2 in [0, 0]
        d3 in [0, 99] d4 in [0, 0] d5 in [0, 0]
        d0 mod 32 in [0, 0]
      )"));
}

TEST_F(ReductionTest, ThreadIndexingOutputLayout) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %input = f32[100,64,512] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64]{0,1} reduce(%input, %c0), dimensions={2}, to_apply=add
    }

    ENTRY entry {
      %input = f32[100,64,512] parameter(0)
      ROOT %fusion = f32[100,64]{0,1} fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5) -> (
          (d3 * 8 + d0 floordiv 32) floordiv 64,
          (d3 * 8 + d0 floordiv 32) mod 64
        )
        domain:
        d0 in [0, 255]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 799]
        d4 in [0, 0]
        d5 in [0, 0]
        d0 mod 32 in [0, 0]
        d3 * 8 + d0 floordiv 32 in [0, 6399]
      )"));
}

TEST_F(ReductionTest, ThreadIndexingSideOutput) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }

    fusion {
      %input = f32[100,64,512] parameter(0)
      %c0 = f32[] constant(0)
      %log = f32[100,64,512] log(%input)
      %reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
      ROOT tuple = (f32[100,64], f32[100,64,512]) tuple(%reduce, %log)
    }

    ENTRY entry {
      %input = f32[100,64,512] parameter(0)
      ROOT %fusion = (f32[100,64], f32[100,64,512]) fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  constexpr char kExpectedIndexing[] = R"(
      (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (
        d3 floordiv 8,
        (d3 mod 8) * 8 + d0 floordiv 32,
        (d0 mod 32) * 2 + s2 * 64 + s3
      )
      domain:
      d0 in [0, 255]
      d1 in [0, 0]
      d2 in [0, 0]
      d3 in [0, 799]
      d4 in [0, 0]
      d5 in [0, 0]
      s0 in [0, 0]
      s1 in [0, 0]
      s2 in [0, 7]
      s3 in [0, 1]
  )";
  auto input_indexing =
      fusion.ComputeThreadIdToInputIndexing(1, 0, &mlir_context_);
  input_indexing->Simplify();
  EXPECT_THAT(input_indexing->ToString(),
              MatchIndexingString(kExpectedIndexing));
  auto output_indexing =
      fusion.ComputeThreadIdToOutputIndexing(1, &mlir_context_);
  output_indexing->Simplify();
  EXPECT_THAT(output_indexing->ToString(),
              MatchIndexingString(kExpectedIndexing));
}

TEST_F(ReductionTest, ThreadIndexingVectorized) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[1024, 8192] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(f32[1024, 8192] %input, f32[] %c0),
        dimensions={1}, to_apply=add
    }
    ENTRY entry {
      %input = f32[1024, 8192] parameter(0)
      ROOT %fusion = f32[1024] fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (
          d3,
          (d0 + s2 * 512) * 2 + s3
        )
        domain:
        d0 in [0, 511]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 1023]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 0]
        s2 in [0, 7]
        s3 in [0, 1]
        d0 + s2 * 512 in [0, 4095]
      )"));
}

TEST_F(ReductionTest, ThreadIndexingBroadcastSideOutput) {
  auto module = ParseAndReturnVerifiedModule(R"(
    %add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    %fusion {
      %p0 = f32[6,6] parameter(0)
      %c0 = f32[] constant(0)
      %reduce = f32[] reduce(%p0, %c0), dimensions={0,1}, to_apply=%add
      %broadcast = f32[6,6] broadcast(%reduce), dimensions={}
      ROOT %tuple = (f32[6,6], f32[]) tuple(%broadcast, %reduce)
    }
    ENTRY main {
      %p0 = f32[6,6] parameter(0)
      ROOT %fusion = (f32[6,6], f32[]) fusion(%p0), kind=kInput, calls=%fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);
  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          (d0 + s2 * 32) floordiv 6,
          (d0 + s2 * 32) mod 6
        )
        domain:
        d0 in [0, 31]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 0]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 0]
        s2 in [0, 15]
        d0 + s2 * 32 in [0, 35]
      )"));
  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> ()
        domain:
        d0 in [0, 31]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 0]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 0]
        s2 in [0, 15]
        (d0 + s2 * 32) mod 6 in [0, 5]
        d0 + s2 * 32 in [0, 35]
      )"));
}

TEST_F(ReductionTest, TwoGroups) {
  auto module = ParseAndReturnVerifiedModule(R"(
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %p0 = f32[2] parameter(0)
      %p1 = f32[2] parameter(1)
      %c0 = f32[] constant(-inf)
      %r0 = f32[] reduce(%p0, %c0), dimensions={0}, to_apply=add
      %c1 = f32[] constant(inf)
      %r1 = f32[] reduce(%p1, %c1), dimensions={0}, to_apply=add
      ROOT %tuple = (f32[], f32[]) tuple(%r0, %r1)
    }
    ENTRY entry {
      %p0 = f32[2] parameter(0)
      %p1 = f32[2] parameter(1)
      ROOT %fusion = (f32[], f32[]) fusion(%p0, %p1), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(fusion.reduction_info().GetGroups().grouped_roots,
              ElementsAre(ElementsAre(&analysis.fusion_root(0).instruction()),
                          ElementsAre(&analysis.fusion_root(1).instruction())));
}

TEST_F(ReductionTest, OneGroup) {
  auto module = ParseAndReturnVerifiedModule(R"(
    %add {
      %p0 = c128[] parameter(0)
      %p1 = c128[] parameter(1)
      ROOT %add.35 = c128[] add(c128[] %p0, c128[] %p1)
    }
    %fusion {
      %p0 = c128[1,2] parameter(0)
      %c0 = c128[] constant((0, 0))
      %reduce = c128[] reduce(%p0, %c0), dimensions={0,1}, to_apply=%add
      %real = f64[] real(c128[] %reduce)
      %imag = f64[] imag(c128[] %reduce)
      %negate = f64[] negate(f64[] %imag)
      ROOT %tuple.29 = (f64[], f64[]) tuple(f64[] %real, f64[] %negate)
    }
    ENTRY entry {
      %p0 = c128[1,2] parameter(0)
      ROOT %fusion = (f64[], f64[]) fusion(%p0), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  FakeReductionFusion fusion(analysis);

  EXPECT_THAT(fusion.reduction_info().GetGroups().grouped_roots, SizeIs(2));

  FakeMlirReductionFusion mlir_fusion(analysis);
  EXPECT_THAT(mlir_fusion.reduction_info().GetGroups().grouped_roots,
              SizeIs(1));
}

TEST_F(ReductionTest, MlirColumnReduction) {
  auto module = ParseAndReturnVerifiedModule(R"(
    add {
      b = f32[] parameter(1)
      a = f32[] parameter(0)
      ROOT out = f32[] add(a, b)
    }
    fusion {
      %p0 = f32[192,64,1536] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[192,1536] reduce(p0, c0), dimensions={1}, to_apply=add
    }
    ENTRY entry {
      %p0 = f32[192,64,1536] parameter(0)
      ROOT %fusion = f32[192,1536] fusion(%p0), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  FakeMlirReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (
          d3 floordiv 12,
          d0 floordiv 32 + s1 * 32,
          ((d3 mod 12) * 32 + d0 mod 32) * 4 + s3
        )
        domain:
        d0 in [0, 1023]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 2303]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 1]
        s2 in [0, 0]
        s3 in [0, 3]
        (d3 mod 12) * 32 + d0 mod 32 in [0, 383]
        d0 floordiv 32 + s1 * 32 in [0, 63]
      )"));
}

TEST_F(ReductionTest, MlirColumnReductionVectorSizeTwo) {
  auto module = ParseAndReturnVerifiedModule(R"(
    add {
      b = f32[] parameter(1)
      a = f32[] parameter(0)
      ROOT out = f32[] add(a, b)
    }
    fusion {
      %p0 = f32[192,64,1538] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[192,1538] reduce(p0, c0), dimensions={1}, to_apply=add
    }
    ENTRY entry {
      %p0 = f32[192,64,1538] parameter(0)
      ROOT %fusion = f32[192,1538] fusion(%p0), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  FakeMlirReductionFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3] -> (
          d3 floordiv 25,
          d0 floordiv 32 + s1 * 32,
          ((d3 mod 25) * 32 + d0 mod 32) * 2 + s3
        )
        domain:
        d0 in [0, 1023]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 4799]
        d4 in [0, 0]
        d5 in [0, 0]
        s0 in [0, 0]
        s1 in [0, 1]
        s2 in [0, 0]
        s3 in [0, 1]
        (d3 mod 25) * 32 + d0 mod 32 in [0, 768]
        d0 floordiv 32 + s1 * 32 in [0, 63]
      )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
