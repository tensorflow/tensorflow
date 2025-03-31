// RUN: triton-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @test_dce_tmem_alloc
//   CHECK-NOT:   ttng.tmem_alloc
//       CHECK:   tt.return
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {
tt.func @test_dce_tmem_alloc(%arg: tensor<128x4xi8, #linear>) {
    %a = ttng.tmem_alloc %arg : (tensor<128x4xi8, #linear>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
    tt.return
}
}  // end module
