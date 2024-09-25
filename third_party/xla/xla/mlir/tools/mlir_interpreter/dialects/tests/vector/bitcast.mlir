// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bitcast1d() -> vector<4xi8> {
  %c = arith.constant dense<-2> : vector<1xi32>
  %b = vector.bitcast %c : vector<1xi32> to vector<4xi8>
  return %b : vector<4xi8>
}

// CHECK-LABEL: @bitcast1d
// CHECK-NEXT: Results
// CHECK-NEXT: vector<4xi8>: [-2, -1, -1, -1]

func.func @bitcast2d() -> vector<2x1xi64> {
  %c = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xi32>
  %b = vector.bitcast %c : vector<2x2xi32> to vector<2x1xi64>
  return %b : vector<2x1xi64>
}

// CHECK-LABEL: @bitcast2d
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x1xi64>: [[4294967296], [12884901890]]

func.func @transpose_bitcast() -> vector<2x1xi64> {
  %c = arith.constant dense<[[0, 1], [2, 3]]> : vector<2x2xi32>
  %d = vector.transpose %c, [1, 0] : vector<2x2xi32> to vector<2x2xi32>
  %b = vector.bitcast %d : vector<2x2xi32> to vector<2x1xi64>
  return %b : vector<2x1xi64>
}

// CHECK-LABEL: @transpose_bitcast
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: vector<2x1xi64>: [[8589934592], [12884901889]]
