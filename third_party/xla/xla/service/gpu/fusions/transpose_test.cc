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
#include "xla/service/gpu/fusions/transpose.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class TransposeTest : public HloTestBase {
 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

absl::StatusOr<std::unique_ptr<TransposeFusion>> GetTransposeFusion(
    const HloFusionAnalysis& analysis) {
  auto emitter = GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis});
  auto fusion = dynamic_cast<TransposeFusion*>(emitter.get());
  TF_RET_CHECK(fusion != nullptr);

  emitter.release();
  return std::unique_ptr<TransposeFusion>{fusion};
}

TEST_F(TransposeTest, ThreadIndexing021) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      %input = f32[100,32,64] parameter(0)
      ROOT transpose = f32[100,64,32] transpose(%input), dimensions={0,2,1}
    }

    ENTRY entry {
      %input = f32[100,32,64] parameter(0)
      ROOT %fusion = f32[100,64,32] fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetTransposeFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          d0 floordiv 32 + s1 * 4,
          (d3 mod 2) * 32 + d0 mod 32
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 200)
        d4 in [0, 1)
        d5 in [0, 1)

        s0 in [0, 1)
        s1 in [0, 8)
        s2 in [0, 1)
      )"));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          (d3 mod 2) * 32 + s1 * 4 + d0 floordiv 32,
          d0 mod 32
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 200)
        d4 in [0, 1)
        d5 in [0, 1)

        s0 in [0, 1)
        s1 in [0, 8)
        s2 in [0, 1)
      )"));
}

TEST_F(TransposeTest, ThreadIndexing201) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      %input = f32[100,64,32] parameter(0)
      ROOT transpose = f32[32,100,64] transpose(%input), dimensions={2,0,1}
    }

    ENTRY entry {
      %input = f32[100,64,32] parameter(0)
      ROOT %fusion = f32[32,100,64] fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetTransposeFusion(analysis));
  mlir::MLIRContext mlir_context;
  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          (d3 * 32 + s1 * 4) mod 64 + d0 floordiv 32,
          d0 mod 32
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 200)
        d4 in [0, 1)
        d5 in [0, 1)

        s0 in [0, 1)
        s1 in [0, 8)
        s2 in [0, 1)
      )"));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d0 floordiv 32 + s1 * 4,
          d3 floordiv 2,
          (d3 mod 2) * 32 + d0 mod 32
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 200)
        d4 in [0, 1)
        d5 in [0, 1)

        s0 in [0, 1)
        s1 in [0, 8)
        s2 in [0, 1)
      )"));
}

TEST_F(TransposeTest, ThreadIndexingPartialBlock) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule m

    fused_computation {
      %p0 = f64[24,2,6,4] parameter(0)
      ROOT %t = f64[6,4,2,24] transpose(%p0), dimensions={2,3,1,0}
    }

    ENTRY main {
      %p0 = f64[24,2,6,4] parameter(0)
      ROOT %fusion = f64[6,4,2,24] fusion(%p0), kind=kInput,
        calls=%fused_computation
    }
  )")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetTransposeFusion(analysis));
  mlir::MLIRContext mlir_context;
  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d0 floordiv 32 + s0 * 4,
          d3,
          (d0 floordiv 4) mod 8,
          d0 mod 4
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 2)
        d4 in [0, 1)
        d5 in [0, 1)
        s0 in [0, 8)
        s1 in [0, 1)
        s2 in [0, 1)
        d0 floordiv 32 + s0 * 4 in [0, 24)
        d0 mod 32 in [0, 24)
      )"));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          s0,
          d0 floordiv 32,
          d3,
          d0 mod 32
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 2)
        d4 in [0, 1)
        d5 in [0, 1)
        s0 in [0, 8)
        s1 in [0, 1)
        s2 in [0, 1)
        d0 floordiv 32 + s0 * 4 in [0, 24)
        d0 mod 32 in [0, 24)
      )"));
}

TEST_F(TransposeTest, SameInputIndexingForRealHeroAndSideOutput) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      %input = f32[100,32,64] parameter(0)
      %transpose = f32[100,64,32] transpose(%input), dimensions={0,2,1}
      %bitcast = f32[100,2048] bitcast(%input)
      ROOT %tuple = (f32[100,64,32], f32[100,2048]) tuple(%transpose, %bitcast)
    }

    ENTRY entry {
      %input = f32[100,32,64] parameter(0)
      ROOT %fusion = (f32[100,64,32], f32[100,2048]) fusion(%input), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetTransposeFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
      fusion->ComputeThreadIdToInputIndexing(1, 0, &mlir_context)->ToString());
}

TEST_F(TransposeTest, ThreadIndexingSideOutput) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule module

    fusion {
      %input0 = f32[100,32,64] parameter(0)
      %input1 = f32[100,32] parameter(1)
      %transpose = f32[100,64,32] transpose(%input0), dimensions={0,2,1}
      %broadcast = f32[100,32,64] broadcast(%input1), dimensions={0,1}
      ROOT %tuple = (f32[100,64,32], f32[100,32,64]) tuple(%transpose, %broadcast)
    }

    ENTRY entry {
      %input0 = f32[100,32,64] parameter(0)
      %input1 = f32[100,32] parameter(1)
      ROOT %fusion = (f32[100,64,32], f32[100,32,64]) fusion(%input0, %input1), kind=kInput, calls=fusion
    })")
                    .value();

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetTransposeFusion(analysis));
  mlir::MLIRContext mlir_context;
  // Check if side output `%broadcast` get the correct input indexing, which
  // should corresponds to `%input1` with shape [100,32].
  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(1, 0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          d0 floordiv 32 + s1 * 4
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 200)
        d4 in [0, 1)
        d5 in [0, 1)

        s0 in [0, 1)
        s1 in [0, 8)
        s2 in [0, 1)
      )"));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(1, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          d3 floordiv 2,
          d0 floordiv 32 + s1 * 4,
          (d3 mod 2) * 32 + d0 mod 32
        )
        domain:
        d0 in [0, 128)
        d1 in [0, 1)
        d2 in [0, 1)
        d3 in [0, 200)
        d4 in [0, 1)
        d5 in [0, 1)

        s0 in [0, 1)
        s1 in [0, 8)
        s2 in [0, 1)
      )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
