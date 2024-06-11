// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
// CHECK: #[[SHARED:.+]] = #triton_gpu.shared
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func @sparse_dot_metadata(%meta: tensor<64x4xi16, #blocked>) {
    // CHECK: %[[META:.+]] = triton_gpu.local_alloc {{.+}} : (tensor<64x4xi16, #blocked>) -> !tt.memdesc<64x4xi16, #[[SHARED]], #triton_gpu.shared_memory>
    // CHECK: triton_gpu.local_load %[[META]] : !tt.memdesc<64x4xi16, #[[SHARED]], #triton_gpu.shared_memory> -> tensor<64x4xi16, #triton_gpu.sparse_dot_meta<{parent = #mma}>>
    %0 = triton_gpu.convert_layout %meta : tensor<64x4xi16, #blocked> -> tensor<64x4xi16, #triton_gpu.sparse_dot_meta<{parent = #mma}>>
    tt.return
  }
}
