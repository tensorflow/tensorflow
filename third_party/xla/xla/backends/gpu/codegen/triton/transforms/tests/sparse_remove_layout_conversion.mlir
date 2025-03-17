// RUN: xla-opt %s --sparse-remove-layout-conversion | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
// CHECK: #[[SHARED:.+]] = #ttg.shared
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @sparse_dot_metadata(%meta: tensor<64x4xi16, #blocked>) {
    // CHECK: %[[META:.+]] = ttg.local_alloc {{.+}} : (tensor<64x4xi16, #blocked>) -> !ttg.memdesc<64x4xi16, #[[SHARED]], #smem>
    // CHECK: ttg.local_load %[[META]] : !ttg.memdesc<64x4xi16, #[[SHARED]], #smem> -> tensor<64x4xi16, #triton_xla.sparse_dot_meta<{parent = #mma}>>
    %0 = ttg.convert_layout %meta : tensor<64x4xi16, #blocked> -> tensor<64x4xi16, #triton_xla.sparse_dot_meta<{parent = #mma}>>
    tt.return
  }
}
