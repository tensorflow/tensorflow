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
#include "tsl/lib/core/status_test_util.h"

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
      param_0 = f32[5,200,2048] parameter(0)
      param_1 = f32[5,200,2048] parameter(1)
      param_2 = f32[] parameter(2)
      ROOT d.1 = (f32[5,200], f32[5,200])
        reduce(param_0, param_1, param_2, param_2), dimensions={2}, to_apply=Add
    }
    ENTRY main {
      a = f32[5, 200, 2048] parameter(0)
      b = f32[5, 200, 2048] parameter(1)
      c = f32[] constant(0)
      ROOT fusion = (f32[5,200], f32[5,200]) fusion(a, b, c),
        kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
// CHECK:      @fused_computation
// CHECK-SAME:   %[[ARG0:.*]]: tensor<5x200x2048xf32> {xla.slice_index = 0
// CHECK-SAME:   %[[ARG1:.*]]: tensor<5x200x2048xf32> {xla.slice_index = 1
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
// CHECK:        %[[A_SHARED:.*]] = xla_gpu.allocate_shared : tensor<2x4xf32>
// CHECK:        %[[B_SHARED:.*]] = xla_gpu.allocate_shared : tensor<2x4xf32>
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
      abs = f32[8] abs(reduce1)
      neg = f32[8] negate(reduce2)
      ROOT tuple = (f32[8], f32[8], f32[8]) tuple(log, neg, abs)
    }
    ENTRY main {
      a = f32[8,2048] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = (f32[8], f32[8], f32[8]) fusion(a, c), kind=kInput,
        calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-DAG: pure_call @Add_add
    // CHECK-DAG: shuffle_reduce @Add_add
    // CHECK-DAG: pure_call @Mul_mul
    // CHECK-DAG: shuffle_reduce @Mul_mul
    // CHECK: allocate_shared
    // CHECK: allocate_shared
    // CHECK: sync_threads
    // CHECK-DAG: shuffle_reduce @Add_add
    // CHECK-DAG: shuffle_reduce @Mul_mul
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, ColumnReduction) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[123,2051,321] parameter(0)
      param_1 = f32[] parameter(1)
      ROOT reduce = f32[123,321] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f32[123,2051,321] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[123,321] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: xla_gpu.pure_call @Add_add
    // CHECK: allocate_shared
    // CHECK: predicated_insert
    // CHECK: sync_threads
    // CHECK: predicated_extract
    // CHECK: shuffle_reduce
    // CHECK: predicated_insert
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, SmallColumnReduction) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[3,128,4] parameter(0)
      param_1 = f32[] parameter(1)
      ROOT reduce = f32[3,4] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f32[3,128,4] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[3,4] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, F64RowReduction) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT add = f64[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f64[100,128] parameter(0)
      param_1 = f64[] parameter(1)
      ROOT reduce = f64[100] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f64[100,128] parameter(0)
      c = f64[] constant(0)
      ROOT fusion = f64[100] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  // This reduction is small enough not to require shared memory.
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-NOT: allocate_shared
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, MultiRowReduction) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[1024,4] parameter(0)
      param_1 = f32[] parameter(1)
      ROOT reduce = f32[1024] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f32[1024,4] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[1024] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  // Multi-row reductions don't use shared memory.
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: shuffle_reduce {{.*}} to 2
    // CHECK-NOT: allocate_shared
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, NonPowerOfTwoRowReduction) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT add = f64[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f64[100,568] parameter(0)
      param_1 = f64[] parameter(1)
      ROOT reduce = f64[100] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f64[100,568] parameter(0)
      c = f64[] constant(0)
      ROOT fusion = f64[100] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: allocate_shared
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, MixedIndexing) {
  constexpr auto kHloString = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %param_0 = f32[64,128] parameter(0)
      %constant_0 = f32[] constant(0)
      %reduce.1 = f32[128] reduce(f32[64,128] %param_0, f32[] %constant_0), dimensions={0}, to_apply=%add
      %neg = f32[64,128] negate(f32[64,128] %param_0)
      %bitcast = f32[8,8,128]{2,1,0} bitcast(f32[64,128] %neg)
      %reduce.2 = f32[128] reduce(f32[8,8,128]{2,1,0} %bitcast, f32[] %constant_0), dimensions={0,1}, to_apply=%add
      ROOT %tuple.12 = (f32[128], f32[128]) tuple(f32[128] %reduce.1, f32[128] %reduce.2)
    }
    ENTRY entry {
      %param_0 = f32[64,128] parameter(0)
      ROOT %fusion = (f32[128], f32[128]) fusion(%param_0), kind=kInput, calls=fusion
    })";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, NonTrivialEpilogue) {
  constexpr auto kHloString = R"(
    HloModule module
    add {
      p0 = f64[] parameter(0)
      p1 = f64[] parameter(1)
      ROOT add = f64[] add(p0, p1)
    }
    fusion {
      %p0 = f64[4] parameter(0)
      %p1 = f64[4] parameter(1)
      %c0 = f64[] constant(-inf)
      %reduce0 = f64[] reduce(p1, c0), dimensions={0}, to_apply=add
      %bc0 = f64[4] broadcast(reduce0), dimensions={}
      %compare0 = pred[4] compare(p1, bc0), direction=EQ
      %c1 = f64[] constant(0)
      %bc1 = f64[4] broadcast(c1), dimensions={}
      %select.3.1 = f64[4] select(compare0, p0, bc1)
      %reduce1 = f64[] reduce(select.3.1, c1), dimensions={0}, to_apply=add
      %convert0 = f64[4] convert(compare0)
      %reduce2 = f64[] reduce(convert0, c1), dimensions={0}, to_apply=add
      ROOT %tuple.1 = (f64[], f64[], f64[]) tuple(%reduce1, reduce0, reduce2)
    }
    ENTRY main {
      %p0 = f64[4] parameter(0)
      %p1 = f64[4] parameter(1)
      ROOT %fusion = (f64[], f64[], f64[]) fusion(%p0, %p1), kind=kInput,
        calls=fusion
    })";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
