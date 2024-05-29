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
      param_0 = f32[2, 3, 2048] parameter(0)
      param_1 = f32[2, 3, 2048] parameter(1)
      param_2 = f32[] parameter(2)
      ROOT d.1 = (f32[2, 3], f32[2, 3])
        reduce(param_0, param_1, param_2, param_2), dimensions={2}, to_apply=Add
    }
    ENTRY main {
      a = f32[2, 3, 2048] parameter(0)
      b = f32[2, 3, 2048] parameter(1)
      c = f32[] constant(0)
      ROOT fusion = (f32[2, 3], f32[2, 3]) fusion(a, b, c),
        kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
// CHECK:      @fused_computation
// CHECK-SAME:   %[[ARG0:.*]]: tensor<2x3x2048xf32> {xla.slice_index = 0
// CHECK-SAME:   %[[ARG1:.*]]: tensor<2x3x2048xf32> {xla.slice_index = 1
// CHECK-SAME:   %[[INIT_TENSOR:.*]]: tensor<f32> {xla.slice_index = 2
// CHECK-SAME:   %[[OUT0:.*]]: tensor<2x3xf32> {xla.slice_index = 3
// CHECK-SAME:   %[[OUT1:.*]]: tensor<2x3xf32> {xla.slice_index = 4
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
      param_0 = f32[8,1024] parameter(0)
      param_1 = f32[] parameter(1)
      reduce1 = f32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Add
      reduce2 = f32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Mul
      log = f32[8] log(reduce1)
      abs = f32[8] abs(reduce1)
      neg = f32[8] negate(reduce2)
      ROOT tuple = (f32[8], f32[8], f32[8]) tuple(log, neg, abs)
    }
    ENTRY main {
      a = f32[8,1024] parameter(0)
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

TEST_F(ReductionTest, RowReduceMOFGroups) {
  constexpr auto kHloString = R"(
    %add_f32 {
      %x = f32[] parameter(0)
      %y = f32[] parameter(1)
      ROOT %add = f32[] add(%x, %y)
    }

    %fused_computation {
      %param0 = f32[1024] parameter(0)
      %param1 = f32[1024] parameter(1)
      %constant0 = f32[] constant(0)
      %reduce1 = f32[] reduce(%param0, %constant0), dimensions={0}, to_apply=%add_f32
      %reduce2 = f32[] reduce(%param1, %constant0), dimensions={0}, to_apply=%add_f32
      ROOT %tuple = (f32[], f32[]) tuple(%reduce1, %reduce2)
    }

    ENTRY %cluster {
      %param0 = f32[1024] parameter(0)
      %param1 = f32[1024] parameter(1)
      ROOT %fusion = (f32[], f32[])
          fusion(%param0, %param1), kind=kInput, calls=%fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: scf.index_switch %block_id_y
    // CHECK: case 1 {
    // CHECK: default {
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
      param_0 = f32[13,1051,321] parameter(0)
      param_1 = f32[] parameter(1)
      ROOT reduce = f32[13,321] reduce(param_0, param_1), dimensions={1}, to_apply=Add
    }
    ENTRY main {
      a = f32[13,1051,321] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[13,321] fusion(a, c), kind=kInput, calls=fused_computation
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
  TF_EXPECT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1)[s0] -> (d0 + s0 * 128 + (d1 mod 64) * 2)>
    // CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> ((d1 mod 64) * 2 + d0 + 512)>
    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
    // CHECK: %[[FULL_TILES:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]]
    // CHECK-NEXT: scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]]
    // CHECK-NOT: scf.if
    // CHECK: xla_gpu.apply_indexing #[[MAP1]](%[[J]] in [0, 1], %thread_id_x in [0, 255])[%[[I]] in [0, 4]]
    // CHECK: scf.for %[[J:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%{{.*}} = %[[FULL_TILES]])
    // CHECK: scf.if
    // CHECK: xla_gpu.apply_indexing #[[MAP2]](%[[J]] in [0, 1], %thread_id_x in [0, 255])
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

TEST_F(ReductionTest, SideOutput) {
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
      exp = f32[8,2048] exponential(param_0)
      reduce = f32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Add
      ROOT t = (f32[8], f32[8,2048]) tuple(reduce, exp)
    }
    ENTRY main {
      a = f32[8,2048] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = (f32[8], f32[8,2048]) fusion(a, c), kind=kInput,
          calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: @fused_computation
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK: %[[SIDE_OUTPUT:.*]] = xla_gpu.pure_call @fused_computation_exp
    // CHECK-NEXT: tensor.insert %[[SIDE_OUTPUT]]
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, UnsignedSideOutput) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true

    Add {
      lhs = u32[] parameter(0)
      rhs = u32[] parameter(1)
      ROOT add = u32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = u32[8,2048] parameter(0)
      param_1 = u32[] parameter(1)
      add = u32[8,2048] add(param_0, param_0)
      reduce = u32[8] reduce(param_0, param_1), dimensions={1}, to_apply=Add
      ROOT t = (u32[8], u32[8,2048]) tuple(reduce, add)
    }
    ENTRY main {
      a = u32[8,2048] parameter(0)
      c = u32[] constant(0)
      ROOT fusion = (u32[8], u32[8,2048]) fusion(a, c), kind=kInput,
          calls=fused_computation
    })";
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, BroadcastSideOutput) {
  constexpr auto kHloString = R"(
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
    })";

  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: @fused_computation
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, VariadicMOF) {
  constexpr auto kHloString = R"(
    %reducer1 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    %reducer2 {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      p2 = f32[] parameter(2)
      p3 = f32[] parameter(3)
      add0 = f32[] add(p0, p2)
      add1 = f32[] add(p1, p3)
      ROOT tuple = (f32[], f32[]) tuple(add0, add1)
    }
    %fusion {
      %p0 = f32[6,6] parameter(0)
      %c0 = f32[] constant(0)
      %neg = f32[6,6] negate(%p0)
      %reduce1 = f32[] reduce(%neg, %c0), dimensions={0,1}, to_apply=%reducer1
      %reduce2 = (f32[], f32[]) reduce(%p0, %p0, %c0, %c0), dimensions={0,1}, to_apply=%reducer2
      ROOT %tuple = (f32[], (f32[], f32[]), f32[6,6]) tuple(%reduce1, %reduce2, %neg)
    }
    ENTRY main {
      %p0 = f32[6,6] parameter(0)
      ROOT %fusion = (f32[], (f32[], f32[]), f32[6,6]) fusion(%p0), kind=kInput, calls=%fusion
    })";

  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: @fused_computation
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(ReductionTest, ColumnReductionVectorization) {
  constexpr auto kHloString = R"(
    HloModule Test, is_scheduled=true
    Add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }
    fused_computation {
      param_0 = f32[2048,16384] parameter(0)
      param_1 = f32[] parameter(1)
      ROOT reduce = f32[16384] reduce(param_0, param_1), dimensions={0}, to_apply=Add
    }
    ENTRY main {
      a = f32[2048,16384] parameter(0)
      c = f32[] constant(0)
      ROOT fusion = f32[16384] fusion(a, c), kind=kInput, calls=fused_computation
    })";
  TF_ASSERT_OK(EmitAndCheckIR(kHloString, R"(
    // CHECK: vector<4xf32>
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
