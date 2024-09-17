// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @insert_1d_1d() -> vector<2xi32> {
  %c = arith.constant dense<[3, 4]> : vector<2xi32>
  %d = arith.constant dense<[1, 2]> : vector<2xi32>
  %i = vector.insert %c, %d[] : vector<2xi32> into vector<2xi32>
  return %i : vector<2xi32>
}

// CHECK-LABEL: @insert_1d_1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<2xi32>: [3, 4]

func.func @insert_1d_0d() -> vector<2xi32> {
  %c = arith.constant 42 : i32
  %d = arith.constant dense<[1, 2]> : vector<2xi32>
  %i = vector.insert %c, %d[1] : i32 into vector<2xi32>
  return %i : vector<2xi32>
}

// CHECK-LABEL: @insert_1d_0d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<2xi32>: [1, 42]

func.func @insert_2d_0d() -> vector<2x2xi32> {
  %c = arith.constant 42 : i32
  %d = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %i = vector.insert %c, %d[0, 1] : i32 into vector<2x2xi32>
  return %i : vector<2x2xi32>
}

// CHECK-LABEL: @insert_2d_0d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x2xi32>: [[1, 42], [3, 4]]

func.func @insert_2d_1d() -> vector<2x3xi32> {
  %c = arith.constant dense<[42, 43, 44]> : vector<3xi32>
  %d = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : vector<2x3xi32>
  %i = vector.insert %c, %d[0] : vector<3xi32> into vector<2x3xi32>
  return %i : vector<2x3xi32>
}

// CHECK-LABEL: @insert_2d_1d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x3xi32>: [[42, 43, 44], [4, 5, 6]]

func.func @insert_2d_2d() -> vector<2x2xi32> {
  %c = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %d = arith.constant dense<0> : vector<2x2xi32>
  %i = vector.insert %c, %d[] : vector<2x2xi32> into vector<2x2xi32>
  return %i : vector<2x2xi32>
}

// CHECK-LABEL: @insert_2d_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x2xi32>: [[1, 2], [3, 4]]

