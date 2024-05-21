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

#include "xla/service/gpu/fusions/concatenate_mlir.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_test_utils.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using MlirConcatenateFusionTest = MlirEmitterTestBase<MlirConcatenateFusion>;

TEST_F(MlirConcatenateFusionTest, ThreadIdIndexing) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

    fused_computation {
      param0 = f32[200] parameter(0)
      param1 = f32[400] parameter(1)
      param2 = f32[300] parameter(2)
      ROOT concat = f32[900] concatenate(param0, param1, param2), dimensions={0}
    }
    ENTRY main {
      param0 = f32[200] parameter(0)
      param1 = f32[400] parameter(1)
      param2 = f32[300] parameter(2)
      ROOT fusion = f32[900] fusion(param0, param1, param2),
        calls=fused_computation, kind=kLoop
    }
  )"));
  thread_id_printer_.SetSymbolName(0, "chunk_id");
  thread_id_printer_.SetSymbolName(1, "unroll_id");

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);
  MlirConcatenateFusion fusion(analysis);

  constexpr auto kIndexing = R"(
    (th_x, th_y, th_z, bl_x, bl_y, bl_z)[chunk_id, unroll_id] -> (
    (bl_x * 128 + th_x) mod 400)
    domain:
    th_x in [0, 127]
    th_y in [0, 0]
    th_z in [0, 0]
    bl_x in [0, 3]
    bl_y in [0, 0]
    bl_z in [0, 0]
    chunk_id in [0, 0]
    unroll_id in [0, 0]
    th_x + bl_x * 128 in [0, 399]
  )";
  auto thread_id_to_output_indexing_0 = fusion.ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/0, &mlir_context_);
  EXPECT_THAT(thread_id_to_output_indexing_0->ToString(thread_id_printer_),
              MatchIndexingString(kIndexing));
  auto thread_id_to_output_indexing_1 = fusion.ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/1, &mlir_context_);
  EXPECT_THAT(thread_id_to_output_indexing_1->ToString(thread_id_printer_),
              MatchIndexingString(kIndexing));
  auto thread_id_to_output_indexing_2 = fusion.ComputeThreadIdToInputIndexing(
      /*root_index=*/0, /*hero_operand_index=*/2, &mlir_context_);
  EXPECT_THAT(thread_id_to_output_indexing_2->ToString(thread_id_printer_),
              MatchIndexingString(kIndexing));
}

TEST_F(MlirConcatenateFusionTest, StandAloneConcatenate) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      param0 = f32[200] parameter(0)
      param1 = f32[400] parameter(1)
      param2 = f32[300] parameter(2)
      ROOT concat = f32[900] concatenate(param0, param1, param2), dimensions={0}
    }
    ENTRY main {
      param0 = f32[200] parameter(0)
      param1 = f32[400] parameter(1)
      param2 = f32[300] parameter(2)
      ROOT fusion = f32[900] fusion(param0, param1, param2),
        calls=fused_computation, kind=kLoop
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-DAG: #[[MAP_1:.*]] = affine_map<(d0, d1) -> ((d1 * 128 + d0) mod 400)>
    // CHECK-DAG: #[[MAP_2:.*]] = affine_map<(d0, d1) -> ((d1 * 128 + d0) mod 400 + 200)>
    // CHECK-DAG: #[[MAP_3:.*]] = affine_map<(d0, d1) -> ((d1 * 128 + d0) mod 400 + 600)>

    // CHECK-LABEL: fused_computation
    // CHECK-SAME:    %[[ARG_0:[a-zA-Z0-9]*]]: {{[^,]*}},
    // CHECK-SAME:    %[[ARG_1:[a-zA-Z0-9]*]]: {{[^,]*}},
    // CHECK-SAME:    %[[ARG_2:[a-zA-Z0-9]*]]: {{[^,]*}},
    // CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]*]]: {{[^,]*}}

    // CHECK: %[[THREAD_ID:.*]] = gpu.thread_id x
    // CHECK: %[[BLOCK_ID:.*]] = gpu.block_id x

    // CHECK: %[[INPUT_INDEX_1:.*]] = xla_gpu.apply_indexing #[[MAP_1]]
    // CHECK: %[[IF_1:.*]] = scf.if
    // CHECK:   %[[VAL_1:.*]] = xla_gpu.pure_call @fused_computation_param0
    // CHECK:   %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1:.*]] into %[[OUTPUT]][%[[INPUT_INDEX_1]]]

    // CHECK: %[[IF_2:.*]] = scf.if
    // CHECK:   %[[VAL_2:.*]] = xla_gpu.pure_call @fused_computation_param1
    // CHECK:   %[[OUTPUT_INDEX_2:.*]] = xla_gpu.apply_indexing #[[MAP_2]]
    // CHECK:   %[[INSERTED_2:.*]] = tensor.insert %[[VAL_2:.*]] into {{.*}}[%[[OUTPUT_INDEX_2]]]

    // CHECK: %[[IF_3:.*]] = scf.if
    // CHECK:   %[[VAL_3:.*]] = xla_gpu.pure_call @fused_computation_param2
    // CHECK:   %[[OUTPUT_INDEX_3:.*]] = xla_gpu.apply_indexing #[[MAP_3]]
    // CHECK:   %[[INSERTED_3:.*]] = tensor.insert %[[VAL_3:.*]] into {{.*}}[%[[OUTPUT_INDEX_3]]]
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, PrologueEpilogue) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      param0 = f32[64] parameter(0)
      param1 = f32[128] parameter(1)
      log = f32[64] log(param0)
      exp = f32[128] exponential(param1)
      concat = f32[192] concatenate(log, exp), dimensions={0}
      ROOT neg = f32[192] negate(concat)
    }
    ENTRY main {
      param0 = f32[64] parameter(0)
      param1 = f32[128] parameter(1)
      ROOT fusion = f32[192] fusion(param0, param1), calls=fused_computation, kind=kLoop
    }
  )";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0 + 64)>

    // CHECK-LABEL: fused_computation
    // CHECK-DAG: %[[C_63:.*]] = arith.constant 63
    // CHECK: %[[THREAD_ID:.*]] = gpu.thread_id x

    // CHECK: %[[IN_BOUND_1:.*]] = arith.cmpi sle, %[[THREAD_ID:.*]], %[[C_63]]
    // CHECK: %[[IF_1:.*]] = scf.if %[[IN_BOUND_1]]
    // CHECK:   %[[VAL_1_1:.*]] = xla_gpu.pure_call @fused_computation_log({{.*}}, %[[THREAD_ID]])
    // CHECK:   %[[VAL_1_2:.*]] = xla_gpu.pure_call @fused_computation__epilogue__neg({{.*}}, %[[THREAD_ID]], %[[VAL_1_1]])
    // CHECK:   %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1_2:.*]] into {{.*}}[%[[THREAD_ID]]]
    // CHECK:   scf.yield %[[INSERTED_1]]

    // CHECK: %[[VAL_2_1:.*]] = xla_gpu.pure_call @fused_computation_exp({{.*}}, %[[THREAD_ID]])
    // CHECK: %[[INDEX_2:.*]] = xla_gpu.apply_indexing #[[MAP]](%[[THREAD_ID]]
    // CHECK: %[[VAL_2_2:.*]] = xla_gpu.pure_call @fused_computation__epilogue__neg({{.*}}, %[[INDEX_2]], %[[VAL_2_1]])
    // CHECK: %[[INSERTED_2:.*]] = tensor.insert %[[VAL_2_2:.*]] into {{.*}}[%[[INDEX_2]]]

    // CHECK: return %[[INSERTED_2]]

    // CHECK: func.func private @fused_computation_log
    // CHECK: func.func private @fused_computation_exp
    // CHECK: func.func private @fused_computation_neg
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, EpilogueSideParameter) {
  auto kHloString = R"(
    HloModule module

    fused_computation {
      param0 = f32[64] parameter(0)
      param1 = f32[192] parameter(1)
      neg = f32[64] negate(param0)
      slice = f32[128] slice(param1), slice={[32:160]}
      exp = f32[128] exponential(slice)
      concat = f32[192] concatenate(neg, exp), dimensions={0}
      ROOT add = f32[192] add(concat, param1)
    }
    ENTRY main {
      param0 = f32[64] parameter(0)
      param1 = f32[192] parameter(1)
      ROOT fusion = f32[192] fusion(param0, param1), calls=fused_computation, kind=kLoop
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, MajorDimension) {
  auto kHloString = R"(
  HloModule module

  fused_computation {
    param0 = f32[16,16] parameter(0)
    param1 = f32[16,16] parameter(1)
    ROOT concat = f32[32,16] concatenate(param0, param1), dimensions={0}
  }
  ENTRY main {
    param0 = f32[16,16] parameter(0)
    param1 = f32[16,16] parameter(1)
    ROOT %fusion = f32[32,16] fusion(param0, param1), kind=kInput, calls=fused_computation
  }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, EpilogueBitcast) {
  auto kHloString = R"(
    HloModule Test

    fused_computation {
      p0 = pred[1] parameter(0)
      p1 = pred[1] parameter(1)
      p2 = pred[1] parameter(2)
      %concatenate.3.3 = pred[3] concatenate(p0, p1, p2), dimensions={0}
      %bitcast.57.1 = pred[1,1,3]{2,1,0} bitcast(pred[3]{0} %concatenate.3.3)
      ROOT %convert.36.1 = u32[1,1,3] convert(pred[1,1,3]{2,1,0} %bitcast.57.1)
    }

    ENTRY main {
      p0 = pred[1] parameter(0)
      p1 = pred[1] parameter(1)
      p2 = pred[1] parameter(2)
      ROOT fusion = u32[1,1,3] fusion(p0, p1, p2), kind=kInput, calls=fused_computation
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
