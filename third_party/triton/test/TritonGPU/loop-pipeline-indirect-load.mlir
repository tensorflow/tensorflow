// RUN: triton-opt %s -tritongpu-pipeline=num-stages=2 | FileCheck %s
// CHECK-LABEL: @indirect_load_two_stages
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: tt.load
// CHECK: async_copy_global_to_local
// CHECK: async_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @indirect_load_two_stages(%arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32, %arg19: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>

    %0 = tt.get_program_id y : i32
    %1 = tt.addptr %arg3, %0 : !tt.ptr<i64>, i32
    %2 = tt.load %1 : !tt.ptr<i64>

    %7 = tt.get_program_id x : i32
    %8 = arith.muli %7, %c16_i32 : i32
    %10 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %15 = tt.splat %8 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %18 = arith.addi %15, %10 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>

    %20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %34 = arith.extsi %arg12 : i32 to i64
    %35 = arith.muli %2, %34 : i64
    %36 = tt.addptr %arg2, %35 : !tt.ptr<f32>, i64

    %47 = tt.splat %arg4 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %48 = tt.addptr %47, %20 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>

    %59 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %61 = arith.extsi %59 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %63 = tt.expand_dims %61 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi64, #blocked3>

    %85 = arith.extsi %22 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> to tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %107 = tt.splat %36 : !tt.ptr<f32> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    %108 = tt.splat %34 : i64 -> tensor<32x1xi64, #blocked3>
    %109 = tt.broadcast %63 : tensor<1x128xi64, #blocked3> -> tensor<32x128xi64, #blocked3>

    %101 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>, #blocked1>
    %111:1 = scf.for %arg28 = %arg18 to %arg19 step %c32_i32 iter_args(%arg29 = %cst) -> (tensor<16x128xf32, #blocked>)  : i32 {
      %129 = tt.splat %arg28 : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %160 = tt.addptr %48, %129 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %161 = tt.load %160 : tensor<32x!tt.ptr<i64>, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %162 = tt.expand_dims %161 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi64, #blocked1>
      %163 = tt.broadcast %162 : tensor<1x32xi64, #blocked1> -> tensor<16x32xi64, #blocked1>
      %182 = tt.addptr %101, %163 : tensor<16x32x!tt.ptr<f32>, #blocked1>, tensor<16x32xi64, #blocked1>
      %183 = tt.load %182 : tensor<16x32x!tt.ptr<f32>, #blocked1>

      %197 = arith.extsi %arg28 : i32 to i64
      %198 = tt.splat %197 : i64 -> tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %199 = arith.addi %198, %85 : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %200 = tt.expand_dims %199 {axis = 1 : i32} : tensor<32xi64, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32x1xi64, #blocked3>
      %201 = arith.muli %200, %108 : tensor<32x1xi64, #blocked3>
      %202 = tt.broadcast %201 : tensor<32x1xi64, #blocked3> -> tensor<32x128xi64, #blocked3>
      %203 = arith.addi %202, %109 : tensor<32x128xi64, #blocked3>
      %204 = tt.addptr %107, %203 : tensor<32x128x!tt.ptr<f32>, #blocked3>, tensor<32x128xi64, #blocked3>
      %209 = tt.load %204 : tensor<32x128x!tt.ptr<f32>, #blocked3>

      %210 = ttg.convert_layout %183 : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %211 = ttg.convert_layout %209 : tensor<32x128xf32, #blocked3> -> tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %212 = tt.dot %210, %211, %arg29 : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x128xf32, #blocked>
      scf.yield %212 : tensor<16x128xf32, #blocked>
    }
    %112 = tt.expand_dims %18 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %113 = tt.splat %2 : i64 -> tensor<16x1xi64, #blocked3>
    %114 = arith.extsi %112 : tensor<16x1xi32, #blocked3> to tensor<16x1xi64, #blocked3>
    %115 = arith.addi %113, %114 : tensor<16x1xi64, #blocked3>
    %116 = arith.extsi %arg17 : i32 to i64
    %117 = tt.splat %116 : i64 -> tensor<16x1xi64, #blocked3>
    %118 = arith.muli %115, %117 : tensor<16x1xi64, #blocked3>
    %119 = tt.expand_dims %59 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3>
    %120 = tt.broadcast %118 : tensor<16x1xi64, #blocked3> -> tensor<16x128xi64, #blocked3>
    %121 = arith.extsi %119 : tensor<1x128xi32, #blocked3> to tensor<1x128xi64, #blocked3>
    %122 = tt.broadcast %121 : tensor<1x128xi64, #blocked3> -> tensor<16x128xi64, #blocked3>
    %123 = arith.addi %120, %122 : tensor<16x128xi64, #blocked3>
    %124 = tt.splat %arg7 : !tt.ptr<f32> -> tensor<16x128x!tt.ptr<f32>, #blocked3>
    %125 = tt.addptr %124, %123 : tensor<16x128x!tt.ptr<f32>, #blocked3>, tensor<16x128xi64, #blocked3>
    %128 = ttg.convert_layout %111#0 : tensor<16x128xf32, #blocked> -> tensor<16x128xf32, #blocked3>
    tt.store %125, %128 : tensor<16x128x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}
