// RUN: mlir-hlo-opt --tile-loops="tile-sizes=2 unroll-factors=4" %s | \
// RUN: FileCheck %s

// CHECK-LABEL: func @parallel_loop
func.func @parallel_loop(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<16xf32>
  scf.parallel (%arg2) = (%c0) to (%c16) step (%c1) {
  // CHECK: %[[C8:.*]] = arith.constant 8
  // CHECK: %[[TILE:.*]] = arith.muli {{.*}} %[[C8]]
  // CHECK: scf.parallel {{.*}} step (%[[TILE]])
  // CHECK:   %[[C4:.*]] = arith.constant 4
  // CHECK:   %[[UNROLL:.*]] = arith.muli {{.*}} %[[C4]]
  // CHECK:   scf.parallel {{.*}} to (%[[TILE]]) step (%[[UNROLL]])
  // CHECK:     scf.parallel
    %2 = memref.load %arg0[%arg2] : memref<16xf32>
    %3 = math.log %2 : f32
    memref.store %3, %0[%arg2] : memref<16xf32>
    scf.yield
  }
  %1 = bufferization.to_tensor %0 : memref<16xf32>
  memref.tensor_store %1, %arg1 : memref<16xf32>
  "lmhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @complex_access
func.func @complex_access(%arg0: memref<16xf32>, %arg1: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  scf.parallel (%arg2) = (%c0) to (%c4) step (%c1) {
  // CHECK:     %[[C2:.*]] = arith.constant 2
  // CHECK:     %[[TILE:.*]] = arith.muli {{.*}} %[[C2]]
  // CHECK:     scf.parallel {{.*}} step (%[[TILE]])
  // CHECK:       scf.parallel
  // We should see only 2 loops for complex access patterns
  // CHECK-NOT:     scf.parallel
    %idx = arith.muli %arg2, %c4 : index
    %2 = memref.load %arg0[%idx] : memref<16xf32>
    %3 = math.log %2 : f32
    memref.store %3, %0[%arg2] : memref<4xf32>
    scf.yield
  }
  %1 = bufferization.to_tensor %0 : memref<4xf32>
  memref.tensor_store %1, %arg1 : memref<4xf32>
  "lmhlo.terminator"() : () -> ()
}