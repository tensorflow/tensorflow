// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @clone() -> (memref<i32>, memref<i32>) {
  %a = arith.constant dense<1> : memref<i32>
  %b = bufferization.clone %a : memref<i32> to memref<i32>
  %c = arith.constant 2 : i32
  memref.store %c, %b[] : memref<i32>
  return %a, %b : memref<i32>, memref<i32>
}

// CHECK-LABEL: @clone
// CHECK-NEXT: Results
// CHECK-NEXT: TensorOrMemref<i32>: 1
// CHECK-NEXT: TensorOrMemref<i32>: 2
