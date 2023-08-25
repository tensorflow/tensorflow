// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @bitcast_scalar() -> i32 {
  %c15 = arith.constant 1.5 : f32
  %ret = arith.bitcast %c15 : f32 to i32
  return %ret : i32
}

// CHECK-LABEL: @bitcast_scalar
// CHECK-NEXT: Results
// CHECK-NEXT: 1069547520

func.func @bitcast_vector() -> vector<2xf32> {
  %c15 = arith.constant dense<[15, 25]> : vector<2xi32>
  %ret = arith.bitcast %c15 : vector<2xi32> to vector<2xf32>
  return %ret : vector<2xf32>
}

// CHECK-LABEL: @bitcast_vector
// CHECK-NEXT: Results
// CHECK-NEXT: [2.101948e-44, 3.503246e-44]
