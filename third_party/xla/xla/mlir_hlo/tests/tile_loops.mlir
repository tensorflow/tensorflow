// RUN: mlir-hlo-opt --tile-loops="tile-sizes=2 unroll-factors=4" %s | \
// RUN: FileCheck %s

// CHECK-LABEL: func @parallel_loop
func.func @parallel_loop(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<16xf32>
  scf.parallel (%arg2) = (%c0) to (%c16) step (%c1) {
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4
  // CHECK:     scf.parallel {{.*}} step (%[[C8]])
  // CHECK:       scf.parallel {{.*}} to (%[[C8]]) step (%[[C4]])
  // CHECK:         scf.parallel {{.*}} to (%[[C4]])
    %2 = memref.load %arg0[%arg2] : memref<16xf32>
    %3 = math.log %2 : f32
    memref.store %3, %0[%arg2] : memref<16xf32>
    scf.reduce
  }
  %1 = bufferization.to_tensor %0 : memref<16xf32> to tensor<16xf32>
  bufferization.materialize_in_destination %1 in writable %arg1
      : (tensor<16xf32>, memref<16xf32>) -> ()
  return
}

// CHECK-LABEL: func @statically_unrolled
func.func @statically_unrolled(%arg0: memref<?xindex>) {
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %c36 = arith.constant 36 : index

  scf.parallel (%arg1) = (%c8) to (%c36) step (%c1) {
  // CHECK: scf.parallel
  // CHECK:   scf.parallel
  // CHECK:     scf.parallel {{.*}} to (%[[C4]])
    memref.store %arg1, %arg0[%arg1] : memref<?xindex>
    scf.reduce
  }
  scf.parallel (%arg1) = (%c0) to (%c36) step (%c3) {
  // CHECK: scf.parallel
  // CHECK:   scf.parallel
  // CHECK:     scf.parallel {{.*}} to (%[[C4]])
    memref.store %arg1, %arg0[%arg1] : memref<?xindex>
    scf.reduce
  }

  return
}

// CHECK-LABEL: func @dynamically_unrolled
func.func @dynamically_unrolled(%arg0: memref<?xindex>, %arg1 : index) {
  // CHECK: %[[C4:.*]] = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c32 = arith.constant 32 : index

  scf.parallel (%arg2) = (%c0) to (%arg1) step (%c1) {
  // CHECK-NOT: scf.parallel {{.*}} to (%[[C4]])
    memref.store %arg2, %arg0[%arg2] : memref<?xindex>
    scf.reduce
  }
  scf.parallel (%arg2) = (%c0) to (%c10) step (%c1) {
  // CHECK-NOT: scf.parallel {{.*}} to (%[[C4]])
    memref.store %arg2, %arg0[%arg2] : memref<?xindex>
    scf.reduce
  }
  scf.parallel (%arg2) = (%c10) to (%c32) step (%c1) {
  // CHECK-NOT: scf.parallel {{.*}} to (%[[C4]])
    memref.store %arg2, %arg0[%arg2] : memref<?xindex>
    scf.reduce
  }
  scf.parallel (%arg2) = (%c0) to (%c32) step (%c10) {
  // CHECK-NOT: scf.parallel {{.*}} to (%[[C4]])
    memref.store %arg2, %arg0[%arg2] : memref<?xindex>
    scf.reduce
  }

  return
}

// CHECK-LABEL: func @complex_access
func.func @complex_access(%arg0: memref<16xf32>, %arg1: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<4xf32>
  scf.parallel (%arg2) = (%c0) to (%c4) step (%c1) {
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK:     scf.parallel {{.*}} step (%[[C2]])
  // CHECK:       scf.parallel
  // We should see only 2 loops for complex access patterns
  // CHECK-NOT:     scf.parallel
    %idx = arith.muli %arg2, %c4 : index
    %2 = memref.load %arg0[%idx] : memref<16xf32>
    %3 = math.log %2 : f32
    memref.store %3, %0[%arg2] : memref<4xf32>
    scf.reduce
  }
  %1 = bufferization.to_tensor %0 : memref<4xf32> to tensor<4xf32>
  bufferization.materialize_in_destination %1 in writable %arg1
      : (tensor<4xf32>, memref<4xf32>) -> ()
  return
}
