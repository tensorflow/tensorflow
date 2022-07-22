// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @memcpy(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @memcpy(%dst: memref<4x4xf32>, %src: memref<4x4xf32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: tfrt_gpu.mem.copy
  "gpu.memcpy"(%dst, %src)
    : (memref<4x4xf32>, memref<4x4xf32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return {{.*}} : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
