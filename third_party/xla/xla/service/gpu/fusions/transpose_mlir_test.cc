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
#include "xla/service/gpu/fusions/transpose_mlir.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class MlirTransposeFusionTest : public MlirEmitterTestBase {
 public:
  std::unique_ptr<MlirFusionEmitterBase> GetEmitter(
      const HloFusionAnalysis& analysis) override {
    return std::make_unique<MlirTransposeFusion>(analysis);
  }
};

TEST_F(MlirTransposeFusionTest, ThreadIndexing021) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      %input = f32[100,32,64] parameter(0)
      ROOT transpose = f32[100,64,32] transpose(%input), dimensions={0,2,1}
    }

    ENTRY entry {
      %input = f32[100,32,64] parameter(0)
      ROOT %fusion = f32[100,64,32] fusion(%input), kind=kInput, calls=fusion
    })"));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  MlirTransposeFusion fusion(analysis);
  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          d0 floordiv 32 + s1 * 4,
          (d3 mod 2) * 32 + d0 mod 32
        )
        domain:
        d0 in [0, 127]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 199]
        d4 in [0, 0]
        d5 in [0, 0]

        s0 in [0, 0]
        s1 in [0, 7]
        s2 in [0, 0]

        (d3 mod 2) * 32 + d0 mod 32 in [0, 63]
        d0 floordiv 32 + s1 * 4 in [0, 31]
      )"));
  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          (d3 mod 2) * 32 + d0 floordiv 32 + s1 * 4,
          d0 mod 32
        )
        domain:
        d0 in [0, 127]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 199]
        d4 in [0, 0]
        d5 in [0, 0]

        s0 in [0, 0]
        s1 in [0, 7]
        s2 in [0, 0]

        (d3 mod 2) * 32 + d0 floordiv 32 + s1 * 4 in [0, 63]
        d0 mod 32 in [0, 31]
      )"));
}

TEST_F(MlirTransposeFusionTest, ThreadIndexing201) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      %input = f32[100,64,32] parameter(0)
      ROOT transpose = f32[32,100,64] transpose(%input), dimensions={2,0,1}
    }

    ENTRY entry {
      %input = f32[100,64,32] parameter(0)
      ROOT %fusion = f32[32,100,64] fusion(%input), kind=kInput, calls=fusion
    })"));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  MlirTransposeFusion fusion(analysis);

  EXPECT_THAT(
      fusion.ComputeThreadIdToInputIndexing(0, 0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          (d3 * 32 + d0 floordiv 32 + s1 * 4) floordiv 64,
          (d3 * 32 + d0 floordiv 32 + s1 * 4) mod 64,
          d0 mod 32
        )
        domain:
        d0 in [0, 127]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 199]
        d4 in [0, 0]
        d5 in [0, 0]

        s0 in [0, 0]
        s1 in [0, 7]
        s2 in [0, 0]

        0 in [0, 0]
        d0 mod 32 in [0, 31]
        d3 * 32 + d0 floordiv 32 + s1 * 4 in [0, 6399]
      )"));
  EXPECT_THAT(
      fusion.ComputeThreadIdToOutputIndexing(0, &mlir_context_)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d0 floordiv 32 + s1 * 4,
          (d3 * 32 + d0 mod 32) floordiv 64,
          (d3 * 32 + d0 mod 32) mod 64
        )
        domain:
        d0 in [0, 127]
        d1 in [0, 0]
        d2 in [0, 0]
        d3 in [0, 199]
        d4 in [0, 0]
        d5 in [0, 0]

        s0 in [0, 0]
        s1 in [0, 7]
        s2 in [0, 0]

        0 in [0, 0]
        d0 floordiv 32 + s1 * 4 in [0, 31]
        d3 * 32 + d0 mod 32 in [0, 6399]
      )"));
}

TEST_F(MlirTransposeFusionTest, FusedTranspose021) {
  auto kHloString = R"(
    HloModule Transpose

    %fused_computation {
      %p0 = f32[20,160,170] parameter(0)
      %exp = f32[20,160,170] exponential(%p0)
      %transpose = f32[20,170,160] transpose(%exp), dimensions={0,2,1}
      ROOT %abs = f32[20,170,160] abs(%transpose)
    }

    ENTRY main {
      %param = f32[20,160,170] parameter(0)
      ROOT %fusion = f32[20,170,160] fusion(%param), kind=kInput, calls=%fused_computation
  })";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
