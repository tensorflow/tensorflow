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

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "tsl/lib/core/status_test_util.h"

namespace xla {
namespace gpu {
namespace {

using MlirConcatenateFusionTest = MlirEmitterTestBase<MlirConcatenateFusion>;

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
    // CHECK: #[[MAP_1:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
    // CHECK: #[[MAP_2:.*]] = affine_map<()[s0, s1] -> ((s0 + s1 * 128) mod 400)>

    // CHECK-LABEL: fused_computation
    // CHECK-SAME:    %[[ARG_0:[a-zA-Z0-9]*]]: {{[^,]*}},
    // CHECK-SAME:    %[[ARG_1:[a-zA-Z0-9]*]]: {{[^,]*}},
    // CHECK-SAME:    %[[ARG_2:[a-zA-Z0-9]*]]: {{[^,]*}},
    // CHECK-SAME:    %[[OUTPUT:[a-zA-Z0-9]*]]: {{[^,]*}}

    // CHECK-DAG: %[[C_200:.*]] = arith.constant 200
    // CHECK-DAG: %[[C_300:.*]] = arith.constant 300
    // CHECK-DAG: %[[C_400:.*]] = arith.constant 400
    // CHECK-DAG: %[[C_600:.*]] = arith.constant 600

    // CHECK: %[[THREAD_ID:.*]] = gpu.thread_id x
    // CHECK: %[[BLOCK_ID:.*]] = gpu.block_id x

    // CHECK: %[[INPUT_INDEX_1:.*]] = affine.apply #[[MAP_2]]()[%[[THREAD_ID]], %[[BLOCK_ID]]]
    // CHECK: %[[IN_BOUND_1:.*]] = arith.cmpi ult, %[[INPUT_INDEX_1:.*]], %[[C_200]]
    // CHECK: %[[IF_1:.*]] = scf.if %[[IN_BOUND_1]]
    // CHECK:   %[[VAL_1:.*]] = xla_gpu.pure_call @fused_computation_param0
    // CHECK:   %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1:.*]] into %[[OUTPUT]][%[[INPUT_INDEX_1]]]

    // CHECK: %[[INPUT_INDEX_2:.*]] = affine.apply #[[MAP_2]]()[%[[THREAD_ID]], %[[BLOCK_ID]]]
    // CHECK: %[[IN_BOUND_2:.*]] = arith.cmpi ult, %[[INPUT_INDEX_2:.*]], %[[C_400]]
    // CHECK: %[[IF_2:.*]] = scf.if %[[IN_BOUND_2]]
    // CHECK:   %[[VAL_2:.*]] = xla_gpu.pure_call @fused_computation_param1
    // CHECK:   %[[OUTPUT_INDEX_2:.*]] = arith.addi %[[INPUT_INDEX_2]], %[[C_200]]
    // CHECK:   %[[INSERTED_2:.*]] = tensor.insert %[[VAL_2:.*]] into {{.*}}[%[[OUTPUT_INDEX_2]]]

    // CHECK: %[[INPUT_INDEX_3:.*]] = affine.apply #[[MAP_2]]()[%[[THREAD_ID]], %[[BLOCK_ID]]]
    // CHECK: %[[IN_BOUND_3:.*]] = arith.cmpi ult, %[[INPUT_INDEX_4:.*]], %[[C_300]]
    // CHECK: %[[IF_3:.*]] = scf.if %[[IN_BOUND_3]]
    // CHECK:   %[[VAL_3:.*]] = xla_gpu.pure_call @fused_computation_param2
    // CHECK:   %[[OUTPUT_INDEX_3:.*]] = arith.addi %[[INPUT_INDEX_3]], %[[C_600]]
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
    // CHECK-LABEL: fused_computation
    // CHECK-DAG: %[[C_64:.*]] = arith.constant 64
    // CHECK: %[[THREAD_ID:.*]] = gpu.thread_id x

    // CHECK: %[[IN_BOUND_1:.*]] = arith.cmpi ult, %[[THREAD_ID:.*]], %[[C_64]]
    // CHECK: %[[IF_1:.*]] = scf.if %[[IN_BOUND_1]]
    // CHECK:   %[[VAL_1_1:.*]] = xla_gpu.pure_call @fused_computation_log({{.*}}, %[[THREAD_ID]])
    // CHECK:   %[[VAL_1_2:.*]] = xla_gpu.pure_call @fused_computation_neg({{.*}}, %[[THREAD_ID]], %[[VAL_1_1]])
    // CHECK:   %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1_2:.*]] into {{.*}}[%[[THREAD_ID]]]
    // CHECK:   scf.yield %[[INSERTED_1]]

    // CHECK: %[[VAL_2_1:.*]] = xla_gpu.pure_call @fused_computation_exp({{.*}}, %[[THREAD_ID]])
    // CHECK: %[[INDEX_2:.*]] = arith.addi %[[THREAD_ID]], %[[C_64]]
    // CHECK: %[[VAL_2_2:.*]] = xla_gpu.pure_call @fused_computation_neg({{.*}}, %[[INDEX_2]], %[[VAL_2_1]])
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

}  // namespace
}  // namespace gpu
}  // namespace xla
