// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @compute_reshape_shape() -> tensor<3xi32> {
  %dynamic_shape = mhlo.constant dense<[2, -1, 3]> : tensor<3xi32>
  %n = arith.constant 24 : index
  %shape = mhlo.compute_reshape_shape %n, %dynamic_shape
    : (index, tensor<3xi32>) -> tensor<3xi32>
  return %shape : tensor<3xi32>
}

// CHECK-LABEL: @compute_reshape_shape
// CHECK-NEXT: Results
// CHECK-NEXT: [2, 4, 3]

func.func @compute_reshape_shape_static() -> tensor<3xi32> {
  %dynamic_shape = mhlo.constant dense<[2, 4, 3]> : tensor<3xi32>
  %n = arith.constant 24 : index
  %shape = mhlo.compute_reshape_shape %n, %dynamic_shape
    : (index, tensor<3xi32>) -> tensor<3xi32>
  return %shape : tensor<3xi32>
}

// CHECK-LABEL: @compute_reshape_shape_static
// CHECK-NEXT: Results
// CHECK-NEXT: [2, 4, 3]

