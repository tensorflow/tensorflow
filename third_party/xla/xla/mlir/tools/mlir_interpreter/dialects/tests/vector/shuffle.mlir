// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @shuffle0d() -> vector<4xi32> {
  %c1 = arith.constant dense<1> : vector<i32>
  %c2 = arith.constant dense<2> : vector<i32>
  %shuffle = vector.shuffle %c1, %c2[0, 1, 1, 0] : vector<i32>, vector<i32>
  return %shuffle : vector<4xi32>
}

// CHECK-LABEL: @shuffle0d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi32>: [1, 2, 2, 1]

func.func @shuffle1d() -> vector<4xi32> {
  %c1 = arith.constant dense<[1, 2]> : vector<2xi32>
  %c2 = arith.constant dense<[11, 12, 13, 14]> : vector<4xi32>
  %shuffle = vector.shuffle %c1, %c2[0, 2, 5, 1] : vector<2xi32>, vector<4xi32>
  return %shuffle : vector<4xi32>
}

// CHECK-LABEL: @shuffle1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi32>: [1, 11, 14, 2]

func.func @shuffle2d() -> vector<3x2xi32> {
  %c1 = arith.constant dense<[[1, 2], [11, 12]]> : vector<2x2xi32>
  %c2 = arith.constant dense<[[21, 22]]> : vector<1x2xi32>
  %shuffle = vector.shuffle %c1, %c2[0, 2, 1] : vector<2x2xi32>, vector<1x2xi32>
  return %shuffle : vector<3x2xi32>
}

// CHECK-LABEL: @shuffle2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<3x2xi32>: [[1, 2], [21, 22], [11, 12]]
