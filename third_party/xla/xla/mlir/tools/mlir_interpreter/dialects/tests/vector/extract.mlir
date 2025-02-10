// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @extract_1d_1d() -> vector<2xi32> {
  %c = arith.constant dense<[1, 2]> : vector<2xi32>
  %i = vector.extract %c[] : vector<2xi32> from vector<2xi32>
  return %i : vector<2xi32>
}

// CHECK-LABEL: @extract_1d_1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<2xi32>: [1, 2]

func.func @extract_1d_0d() -> i32 {
  %c = arith.constant dense<[1, 2]> : vector<2xi32>
  %i = vector.extract %c[1] : i32 from vector<2xi32>
  return %i : i32
}

// CHECK-LABEL: @extract_1d_0d
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 2

func.func @extract_2d_0d() -> i32 {
  %c = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %i = vector.extract %c[0, 1] : i32 from vector<2x2xi32>
  return %i : i32
}

// CHECK-LABEL: @extract_2d_0d
// CHECK-NEXT: Results
// CHECK-NEXT: i32: 2

func.func @extract_2d_1d() -> vector<3xi32> {
  %c = arith.constant dense<[[1, 2, 3], [4, 5, 6]]> : vector<2x3xi32>
  %i = vector.extract %c[0] : vector<3xi32> from vector<2x3xi32>
  return %i : vector<3xi32>
}

// CHECK-LABEL: @extract_2d_1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<3xi32>: [1, 2, 3]

func.func @extract_2d_2d() -> vector<2x2xi32> {
  %c = arith.constant dense<[[1, 2], [3, 4]]> : vector<2x2xi32>
  %i = vector.extract %c[] : vector<2x2xi32> from vector<2x2xi32>
  return %i : vector<2x2xi32>
}

// CHECK-LABEL: @extract_2d_2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x2xi32>: [[1, 2], [3, 4]]

