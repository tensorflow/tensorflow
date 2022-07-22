// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK:      func @replica_id(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @replica_id(%output: memref<ui32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN:%[0-9]+]] = xlir.replica_id %arg1, %arg2, %arg0

  "lmhlo.replica_id"(%output) : (memref<ui32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @partition_id(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func.func @partition_id(%output: memref<ui32>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[CHAIN:%[0-9]+]] = xlir.partition_id %arg1, %arg2, %arg0

  "lmhlo.partition_id"(%output) : (memref<ui32>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
