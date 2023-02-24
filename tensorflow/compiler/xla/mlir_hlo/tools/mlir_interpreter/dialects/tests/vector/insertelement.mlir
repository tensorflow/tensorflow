// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @insert0d() -> vector<i32> {
  %c = arith.constant dense<1> : vector<i32>
  %v = arith.constant 42 : i32
  %i = vector.insertelement %v, %c[] : vector<i32>
  return %i : vector<i32>
}

// CHECK-LABEL: @insert0d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<i32>: 42

func.func @insert1d() -> vector<2xi32> {
  %c = arith.constant dense<1> : vector<2xi32>
  %c1 = arith.constant 1 : index
  %v = arith.constant 42 : i32
  %i = vector.insertelement %v, %c[%c1 : index] : vector<2xi32>
  return %i : vector<2xi32>
}

// CHECK-LABEL: @insert1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<2xi32>: [1, 42]
