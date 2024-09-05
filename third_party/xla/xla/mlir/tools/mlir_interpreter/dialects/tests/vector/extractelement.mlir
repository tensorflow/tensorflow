// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @extract0d() -> i32 {
  %c = arith.constant dense<1> : vector<i32>
  %i = vector.extractelement %c[] : vector<i32>
  return %i : i32
}

// CHECK-LABEL: @extract0d
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 1

func.func @extract1d() -> i32 {
  %c = arith.constant dense<[1,2]> : vector<2xi32>
  %c1 = arith.constant 1 : index
  %i = vector.extractelement %c[%c1 : index] : vector<2xi32>
  return %i : i32
}

// CHECK-LABEL: @extract1d
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 2