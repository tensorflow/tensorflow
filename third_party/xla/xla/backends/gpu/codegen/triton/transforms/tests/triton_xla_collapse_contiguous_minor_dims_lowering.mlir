// RUN: xla-opt %s --triton-xla-collapse-contiguous-minor-dims --triton-xla-extract-insert-to-triton | FileCheck %s

#indexing_map = #xla.indexing_map<"(pid_0) -> (pid_0 * 64), domain: pid_0 in [0, 15]">
module {
  // CHECK-LABEL: tt.func @apply_indexing_lowering
  // CHECK-NOT:   arith.cmpi
  // CHECK:        %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x!tt.ptr<f32>>
  // CHECK:        tt.store %{{.*}}, %{{.*}} : tensor<128x!tt.ptr<f32>>
  // CHECK:        tt.return
  func.func @apply_indexing_lowering(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.index_cast %1 : i64 to index
    %3 = xla.apply_indexing #indexing_map(%2)
    %t = triton_xla.extract from %arg0
        as memref<1024x2xf32, #xtile.layout<[1, 0]>>
        [%3, 0] [64, 2] [1, 1] : tensor<64x2xf32>
    triton_xla.insert %t into %arg1
        as memref<1024x2xf32, #xtile.layout<[1, 0]>>
        [%3, 0] [64, 2] [1, 1] : tensor<64x2xf32>
    func.return
  }
}
