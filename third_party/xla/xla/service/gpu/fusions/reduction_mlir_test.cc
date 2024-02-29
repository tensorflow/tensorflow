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
#include "xla/service/gpu/fusions/reduction_mlir.h"

#include <optional>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/tests/filecheck.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ReductionTest = MlirEmitterTestBase<MlirReductionFusion>;

TEST_F(ReductionTest, VariadicRowReduce) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      scalar_lhs.1 = f32[] parameter(2)
      scalar_rhs.1 = f32[] parameter(3)
      add.0 = f32[] add(scalar_lhs.0, scalar_lhs.1)
      add.1 = f32[] add(scalar_rhs.0, scalar_rhs.1)
      ROOT t = (f32[], f32[]) tuple(add.0, add.1)
    }
    fused_computation {
      param_0 = f32[5,200,300] parameter(0)
      param_1 = f32[5,200,300] parameter(1)
      param_2 = f32[] parameter(2)
      ROOT d.1 = (f32[5,200], f32[5,200])
        reduce(param_0, param_1, param_2, param_2), dimensions={2}, to_apply=Add
    }
    ENTRY main {
      a = f32[5, 200, 300] parameter(0)
      b = f32[5, 200, 300] parameter(1)
      c = f32[] constant(0)
      ROOT fusion = (f32[5,200], f32[5,200]) fusion(a, b, c),
        kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
// CHECK:      @fused_computation
// CHECK-SAME:   %[[ARG0:.*]]: tensor<5x200x300xf32> {xla.slice_index = 0
// CHECK-SAME:   %[[ARG1:.*]]: tensor<5x200x300xf32> {xla.slice_index = 1
// CHECK-SAME:   %[[INIT_TENSOR:.*]]: tensor<f32> {xla.slice_index = 2
// CHECK-SAME:   %[[OUT0:.*]]: tensor<5x200xf32> {xla.slice_index = 3
// CHECK-SAME:   %[[OUT1:.*]]: tensor<5x200xf32> {xla.slice_index = 4
// CHECK:        %[[INIT:.*]] = xla_gpu.pure_call @fused_computation_param_2
// CHECK:        %[[PER_THREAD:.*]]:2 = scf.for
// CHECK-SAME:       iter_args(%[[A:.*]] = %[[INIT]], %[[B:.*]] = %[[INIT]])
// CHECK:          %[[A2:.*]] = xla_gpu.pure_call @fused_computation_param_0
// CHECK:          %[[B2:.*]] = xla_gpu.pure_call @fused_computation_param_1
// CHECK:          xla_gpu.pure_call @Add_t(%[[A]], %[[B]], %[[A2]], %[[B2]])
// CHECK:        %[[SHUFFLED:.*]]:2 = xla_gpu.shuffle_reduce
// CHECK-SAME:     @Add_t(%[[PER_THREAD]]#0, %[[PER_THREAD]]#1) to 16
// CHECK:        %[[A_SHARED:.*]] = xla_gpu.allocate_shared : tensor<8x1xf32>
// CHECK:        %[[B_SHARED:.*]] = xla_gpu.allocate_shared : tensor<8x1xf32>
// CHECK:        predicated_insert %[[SHUFFLED]]#0 into %[[A_SHARED]]
// CHECK:        predicated_insert %[[SHUFFLED]]#1 into %[[B_SHARED]]
// CHECK:        sync_threads
// CHECK-NOT:    shuffle_reduce)
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, RowReduceEpilogue) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[8,2048] parameter(0)
      param_1 = f32[] parameter(1)
      reduce = f32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Add
      ROOT log = f32[8] log(reduce)
    }
    ENTRY main {
      a = f32[8,2048] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[8] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: pure_call @Add_add
    // CHECK: shuffle_reduce
    // CHECK: allocate_shared
    // CHECK: sync_threads
    // CHECK: shuffle_reduce
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, RowReduceMOFEpilogue) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    Mul {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT mul = f32[] multiply(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[8,2048] parameter(0)
      param_1 = f32[] parameter(1)
      reduce1 = f32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Add
      reduce2 = f32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Mul
      log = f32[8] log(reduce1)
      neg = f32[8] negate(reduce2)
      ROOT tuple = (f32[8], f32[8]) tuple(log, neg)
    }
    ENTRY main {
      a = f32[8,2048] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = (f32[8], f32[8]) fusion(a, c), kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: pure_call @Add_add
    // CHECK: shuffle_reduce
    // CHECK: allocate_shared
    // CHECK: pure_call @Mul_mul
    // CHECK: shuffle_reduce
    // CHECK: allocate_shared
    // CHECK: sync_threads
    // CHECK: shuffle_reduce
    // CHECK: shuffle_reduce
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
