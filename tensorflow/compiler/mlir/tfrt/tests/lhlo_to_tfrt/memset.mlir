// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @memset(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME:   %arg3: f32
// CHECK-SAME: ) -> !tfrt.chain
func.func @memset(%dst: memref<4x4xf32>, %value: f32) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: tfrt_gpu.mem.set
  "gpu.memset"(%dst, %value)
    : (memref<4x4xf32>, f32) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return {{.*}} : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
