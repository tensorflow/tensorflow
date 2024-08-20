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
#include "xla/service/gpu/fusions/loop_mlir.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using MlirLoopFusionTest = MlirEmitterTestBase<MlirLoopFusion>;

TEST_F(MlirLoopFusionTest, ThreadId_Broadcast) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    bcast {
      %input = f32[20] parameter(0)
      ROOT bcast = f32[10, 20, 30] broadcast(%input), dimensions={1}
    }
    ENTRY entry {
      %input = f32[20] parameter(0)
      ROOT %fusion = f32[10, 20, 30] fusion(%input), kind=kLoop, calls=bcast
    }
  )"));
  thread_id_printer_.SetSymbolName(0, "chunk_id");
  thread_id_printer_.SetSymbolName(1, "unroll_id");

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = HloFusionAnalysis::Create(*root, device_info_);

  MlirLoopFusion fusion(analysis);
  auto thread_id_to_output_indexing =
      fusion.ComputeThreadIdToOutputIndexing(/*root_index=*/0, &mlir_context_);
  EXPECT_THAT(thread_id_to_output_indexing->ToString(thread_id_printer_),
              MatchIndexingString(R"(
              (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] -> (
                  (bl_x * 128 + th_x) floordiv 600,
                  ((bl_x * 128 + th_x) floordiv 30) mod 20,
                  (bl_x * 128 + th_x) mod 30
                )
                domain:
                th_x in [0, 127]
                th_y in [0, 0]
                th_z in [0, 0]
                bl_x in [0, 46]
                bl_y in [0, 0]
                bl_z in [0, 0]
                chunk_id in [0, 0]
                unroll_id in [0, 0]
                bl_x * 128 + th_x in [0, 5999]
            )"));
  auto thread_id_to_input_indexing = fusion.ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/0, &mlir_context_);
  EXPECT_THAT(thread_id_to_input_indexing->ToString(thread_id_printer_),
              MatchIndexingString(R"(
              (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] ->
                (((bl_x * 128 + th_x) floordiv 30) mod 20)
                domain:
                th_x in [0, 127]
                th_y in [0, 0]
                th_z in [0, 0]
                bl_x in [0, 46]
                bl_y in [0, 0]
                bl_z in [0, 0]
                chunk_id in [0, 0]
                unroll_id in [0, 0]
                bl_x * 128 + th_x in [0, 5999]
            )"));
}

TEST_F(MlirLoopFusionTest, NoCodeDuplication) {
  // This test HLO is copied from
  // xla/service/fusion_node_indexing_evaluation_test.cc.
  auto kHloString = R"(
    HloModule test_module

    %fused_computation (param: f32[6]) -> f32[2] {
      %param = f32[6]{0} parameter(0)
      %slice0.1 = f32[5]{0} slice(f32[6]{0} %param), slice={[0:5]}
      %slice0.2 = f32[5]{0} slice(f32[6]{0} %param), slice={[1:6]}
      %add0 = f32[5]{0} add(f32[5]{0} %slice0.1, f32[5]{0} %slice0.2)
      %slice1.1 = f32[4]{0} slice(f32[5]{0} %add0), slice={[0:4]}
      %slice1.2 = f32[4]{0} slice(f32[5]{0} %add0), slice={[1:5]}
      %add1 = f32[4]{0} add(f32[4]{0} %slice1.1, f32[4]{0} %slice1.2)
      %slice2.1 = f32[3]{0} slice(f32[4]{0} %add1), slice={[0:3]}
      %slice2.2 = f32[3]{0} slice(f32[4]{0} %add1), slice={[1:4]}
      %add2 = f32[3]{0} add(f32[3]{0} %slice2.1, f32[3]{0} %slice2.2)
      %slice3.1 = f32[2]{0} slice(f32[3]{0} %add2), slice={[0:2]}
      %slice3.2 = f32[2]{0} slice(f32[3]{0} %add2), slice={[1:3]}
      ROOT %add3 = f32[2]{0} add(f32[2]{0} %slice3.1, f32[2]{0} %slice3.2)
    }
    ENTRY entry_computation {
      p0 = f32[] parameter(0)
      add = f32[] add(p0, p0)
      broadcast = f32[6]{0} broadcast(add), dimensions={}
      ROOT %fusion = f32[2]{0} fusion(broadcast), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-COUNT-4: arith.add
    // CHECK-NOT: arith.add
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirLoopFusionTest, IotaCopyBitcastBroadcastReshapeReverseTranspose) {
  auto kHloString = R"(
    HloModule test_module

    %fused_computation {
      %iota = f32[10,20,30] iota(), iota_dimension=2
      %copy = f32[10,20,30] copy(%iota)
      %bitcast = s32[10,20,30] bitcast-convert(%copy)
      %broadcast = s32[2,10,3,20,5,30,7] broadcast(%bitcast),
        dimensions={1,3,5}
      %reshape = s32[20,60,150,7] reshape(%broadcast)
      %reverse = s32[20,60,150,7] reverse(%reshape), dimensions={2,3}
      ROOT %transpose = s32[60,20,7,150] transpose(%reverse),
        dimensions={1,0,3,2}
    }
    ENTRY entry_computation {
      ROOT %fusion = s32[60,20,7,150] fusion(),
        kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-COUNT-2: func.func
    // CHECK-NOT:     func.func
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirLoopFusionTest, DynamicSliceWith64BitInput) {
  // Lowering this kernel with 32 bit indices causes an underflow of `c`,
  // resulting in slicing the last four elements instead of the first four.
  constexpr auto kHloString = R"(
    %fused_computation {
      %p0 = s64[] parameter(0)
      %p1 = f64[5] parameter(1)
      ROOT slice = f64[4] dynamic-slice(%p1, %p0), dynamic_slice_sizes={4}
    }

    ENTRY main {
      %c = s64[] constant(-1000000000000)
      %p0 = f64[5] parameter(0)
      ROOT %fusion = f64[4]{0} fusion(%c, %p0), kind=kInput, calls=%fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i32>>
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
