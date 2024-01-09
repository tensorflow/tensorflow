// RUN: mlir-hlo-opt --unroll-loops --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @unroll
func.func @unroll(%arg0: memref<11xf32>) {
  // CHECK: %[[ZERO:.*]] = arith.constant 0.0
  %zero = arith.constant 0.0 : f32

  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c13 = arith.constant 13 : index

  // CHECK: memref.store %[[ZERO]], %arg0[%c1]
  // CHECK: memref.store %[[ZERO]], %arg0[%c4]
  // CHECK: memref.store %[[ZERO]], %arg0[%c7]
  // CHECK: memref.store %[[ZERO]], %arg0[%c10]
  scf.for %i = %c1 to %c13 step %c3 {
    memref.store %zero, %arg0[%i] : memref<11xf32>
  }

  // CHECK-NEXT: return
  return
}
