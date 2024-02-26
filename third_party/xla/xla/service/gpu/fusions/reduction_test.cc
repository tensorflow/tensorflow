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
#include "xla/service/gpu/fusions/reduction.h"

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class ReductionTest : public HloTestBase {
 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
};

absl::StatusOr<std::unique_ptr<ReductionFusion>> GetReductionFusion(
    const HloFusionAnalysis& analysis) {
  TF_ASSIGN_OR_RETURN(
      auto emitter, GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis}));
  auto fusion = dynamic_cast<ReductionFusion*>(emitter.get());
  TF_RET_CHECK(fusion != nullptr);

  emitter.release();
  return std::unique_ptr<ReductionFusion>{fusion};
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

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetReductionFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
      MatchIndexingString(R"(
        (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
          (d3 * 8 + d0 floordiv 32) floordiv 64,
          (d3 * 8 + d0 floordiv 32) mod 64,
          d0 mod 32 + s2 * 32
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
        s2 in [0, 15]
        0 in [0, 0]
        d0 mod 32 + s2 * 32 in [0, 511]
        d3 * 8 + d0 floordiv 32 in [0, 6399]
      )"));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
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
        (d3 * 8 + d0 floordiv 32) mod 64 in [0, 63]
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

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetReductionFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
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
        0 in [0, 0]
        d0 mod 4 in [0, 3]
        d3 * 64 + d0 floordiv 4 in [0, 6399]
      )"));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
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
        (d0 floordiv 4) mod 64 in [0, 63]
        d0 mod 4 in [0, 0]
        d3 * 64 + d0 floordiv 4 in [0, 6399]
        d3 + (d0 floordiv 4) floordiv 64 in [0, 99]
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

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetReductionFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
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
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
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

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetReductionFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(0, &mlir_context)->ToString(),
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
        (d3 * 8 + d0 floordiv 32) mod 64 in [0, 63]
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

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetReductionFusion(analysis));
  mlir::MLIRContext mlir_context;

  constexpr char kExpectedIndexing[] = R"(
      (d0, d1, d2, d3, d4, d5)[s0, s1, s2] -> (
        (d3 * 8 + d0 floordiv 32) floordiv 64,
        (d3 * 8 + d0 floordiv 32) mod 64,
        d0 mod 32 + s2 * 32
      )
      domain:
      d0 in [0, 255]
      d1 in [0, 0]
      d2 in [0, 0]
      d3 in [0, 799]
      d4 in [1, 0]
      d5 in [0, 0]
      s0 in [0, 0]
      s1 in [0, 0]
      s2 in [0, 15]
      0 in [0, 0]
      d0 mod 32 + s2 * 32 in [0, 511]
      d3 * 8 + d0 floordiv 32 in [0, 6399]
  )";
  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(1, 0, &mlir_context)->ToString(),
      MatchIndexingString(kExpectedIndexing));
  EXPECT_THAT(
      fusion->ComputeThreadIdToOutputIndexing(1, &mlir_context)->ToString(),
      MatchIndexingString(kExpectedIndexing));
}

TEST_F(ReductionTest, bla) {
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

  TF_ASSERT_OK_AND_ASSIGN(auto fusion, GetReductionFusion(analysis));
  mlir::MLIRContext mlir_context;

  EXPECT_THAT(
      fusion->ComputeThreadIdToInputIndexing(0, 0, &mlir_context)->ToString(),
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
        0 in [0, 0]
        d0 + s2 * 512 in [0, 4095]
      )"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
