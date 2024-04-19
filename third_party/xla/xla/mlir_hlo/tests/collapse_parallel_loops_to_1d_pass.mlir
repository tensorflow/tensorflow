// RUN: mlir-hlo-opt --collapse-parallel-loops-to-1d %s | \
// RUN: FileCheck %s

// CHECK-LABEL: func @parallel_2d
func.func @parallel_2d(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.alloc() {alignment = 128 : i64} : memref<4x4xf32>
  scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c4, %c4) step (%c1, %c1) {
  // CHECK: scf.parallel ({{[^.]+}})
    %2 = memref.load %arg0[%arg2,%arg3] : memref<4x4xf32>
    %3 = math.log %2 : f32
    memref.store %3, %0[%arg2,%arg3] : memref<4x4xf32>
    scf.reduce
  }
  %1 = bufferization.to_tensor %0 : memref<4x4xf32>
  bufferization.materialize_in_destination %1 in writable %arg1
      : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}